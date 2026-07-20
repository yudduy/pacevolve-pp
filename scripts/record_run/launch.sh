#!/bin/bash
# Mac-side launcher for the record run: create pod (with terminate-after stamp)
# -> wait ports (NBSP-safe) -> push payload (repo subset + ttt_discover pkg +
# minimal .env) -> launch server bootstrap AND pod chain -> write state file.
# After LAUNCH-COMPLETE the pod is fully autonomous; the Mac is optional.
# Usage: mac_launch_record_run.sh 32b|8b <RID>
set -u
SIZE="${1:?32b|8b}"; RID="${2:?rid}"
SP="$(cd "$(dirname "$0")" && pwd)"
PP=/Users/c-dnguyen/Documents/project/pacevolve-pp
TTT=/Users/c-dnguyen/Documents/project/ttt-discover
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
case "$SIZE" in
  32b) HF="Qwen/Qwen3-32B"; GPUS=4; BOOT="$SP/bootstrap_32b_tp4.sh"; HOURS=16;;
  8b)  HF="Qwen/Qwen3-8B";  GPUS=1; BOOT="$SP/bootstrap_8b.sh";      HOURS=22;;
  *) echo "LAUNCH-FAIL bad size"; exit 2;;
esac
NAME="rfg-record-${SIZE}-${RID}"
TERM_AT=$(date -u -v+"${HOURS}"H +%Y-%m-%dT%H:%M:%SZ)
echo "=== create $NAME gpus=$GPUS terminate_after=$TERM_AT ==="
# Capacity ladder: same-class 80GB cards, cheapest viable first. SXM beats PCIe
# for TP4 (NVLink all-reduce); secure tier is the paid fallback when community
# stock is dry. --public-ip is a community-only flag.
POD=""; ATTEMPT=0
while [ -z "$POD" ] && [ "$ATTEMPT" -lt 20 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  for SPEC in "NVIDIA A100 80GB PCIe|COMMUNITY|--public-ip" \
              "NVIDIA A100-SXM4-80GB|COMMUNITY|--public-ip" \
              "NVIDIA A100 80GB PCIe|SECURE|" \
              "NVIDIA A100-SXM4-80GB|SECURE|"; do
    GPUID="${SPEC%%|*}"; REST="${SPEC#*|}"; CLOUD="${REST%%|*}"; PUBIP="${REST#*|}"
    echo "--- attempt $ATTEMPT: $GPUID / $CLOUD"
    OUT=$(runpodctl pod create --name "$NAME" \
      --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
      --gpu-id "$GPUID" --gpu-count "$GPUS" --cloud-type "$CLOUD" \
      --container-disk-in-gb 300 --ports "8021/tcp,22/tcp" $PUBIP \
      --terminate-after "$TERM_AT" 2>&1)
    if echo "$OUT" | grep -q '"error"'; then
      echo "$OUT" | grep '"error"' | head -1
      continue
    fi
    echo "$OUT" | head -3
    POD=$(echo "$OUT" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('{'):
        try:
            d = json.loads(line)
            if isinstance(d, dict) and d.get('id'):
                print(d['id']); break
        except Exception:
            pass" 2>/dev/null)
    [ -z "$POD" ] && POD=$(echo "$OUT" | grep -oE '"[a-z0-9]{12,16}"' | head -1 | tr -d '"')
    [ -z "$POD" ] && POD=$(runpodctl pod list 2>/dev/null | grep "$NAME" | grep -oE '^[a-z0-9]+' | head -1)
    if [ -n "$POD" ]; then echo "CREATED-ON $GPUID / $CLOUD"; break; fi
  done
  [ -z "$POD" ] && sleep 90
done
[ -z "$POD" ] && { echo "LAUNCH-FAIL no-capacity-after-${ATTEMPT}-rounds"; exit 1; }
echo "POD=$POD"
FAIL() { echo "LAUNCH-FAIL $1; removing pod $POD"; runpodctl remove pod "$POD" >/dev/null 2>&1; exit 1; }

# ports (runpodctl tables pad with NBSP — never anchor on ASCII space)
DEADLINE=$((SECONDS + 900)); SSH_EP=""; API_EP=""
while [ $SECONDS -lt $DEADLINE ]; do
  PORTS=$(runpodctl get pod "$POD" -a 2>/dev/null | tail -1)
  SSH_EP=$(echo "$PORTS" | grep -oE "[0-9.]+:[0-9]+->22[^0-9]" | head -1 | sed -E "s/->22[^0-9]?//")
  API_EP=$(echo "$PORTS" | grep -oE "[0-9.]+:[0-9]+->8021[^0-9]" | head -1 | sed -E "s/->8021[^0-9]?//")
  [ -n "$SSH_EP" ] && [ -n "$API_EP" ] && break
  sleep 20
done
if [ -z "$SSH_EP" ] || [ -z "$API_EP" ]; then
  echo "LAUNCH-FAIL ports-never-assigned; removing pod $POD"
  runpodctl remove pod "$POD"
  exit 1
fi
SSH_IP="${SSH_EP%%:*}"; SSH_PORT="${SSH_EP##*:}"
API_IP="${API_EP%%:*}"; API_PORT="${API_EP##*:}"
echo "ENDPOINTS ssh=$SSH_IP:$SSH_PORT api=$API_IP:$API_PORT"

SSHQ() { ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i "$KEY" -p "$SSH_PORT" "root@$SSH_IP" "$@"; }
DEADLINE=$((SECONDS + 900))
until SSHQ 'echo pod-up' 2>/dev/null | grep -q pod-up; do
  [ $SECONDS -ge $DEADLINE ] && { echo "LAUNCH-FAIL ssh-never-up"; exit 1; }
  sleep 20
done
echo "=== payload (tar over ssh — RunPod secure images lack rsync) ==="
tar -C "$PP" --exclude='.git' --exclude='.venv' --exclude='.env' --exclude='results' \
  --exclude='__pycache__' --exclude='wandb' -czf - . \
  | SSHQ 'mkdir -p /workspace/pp && tar -xzf - -C /workspace/pp' || FAIL tar-pp
tar -C "$TTT" --exclude='__pycache__' -czf - ttt_discover \
  | SSHQ 'mkdir -p /workspace/ttt && tar -xzf - -C /workspace/ttt' || FAIL tar-ttt
ENVTMP="$SP/env.pod.$$"
grep -E '^(OPENROUTER_API_KEY|OPENROUTER_BASE_URL|WANDB_API_KEY|WANDB_ENTITY|WANDB_PROJECT)=' "$PP/.env" > "$ENVTMP"
scp -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i "$KEY" -P "$SSH_PORT" \
  "$ENVTMP" "root@$SSH_IP:/workspace/pp/.env" || { rm -f "$ENVTMP"; FAIL scp-env; }
rm -f "$ENVTMP"
scp -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i "$KEY" -P "$SSH_PORT" \
  "$BOOT" "root@$SSH_IP:/workspace/bootstrap.sh"
scp -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i "$KEY" -P "$SSH_PORT" \
  "$SP/pod_chain.sh" "$SP/pod_reaper.sh" "$SP/upload_artifact.py" "root@$SSH_IP:/workspace/"

echo "=== launch bootstrap + chain + reaper ==="
SSHQ "nohup bash /workspace/bootstrap.sh > /workspace/bootstrap.log 2>&1 & sleep 1; nohup bash /workspace/pod_chain.sh $RID $HF 4 128 > /workspace/chain.log 2>&1 & sleep 1; nohup bash /workspace/pod_reaper.sh $RID > /dev/null 2>&1 & sleep 1; echo launched-all" \
  || FAIL remote-launch

cat > "$SP/record_run.state" <<EOF
POD=$POD
RID=$RID
SIZE=$SIZE
HF=$HF
SSH_IP=$SSH_IP
SSH_PORT=$SSH_PORT
API_IP=$API_IP
API_PORT=$API_PORT
TERM_AT=$TERM_AT
EOF
echo "LAUNCH-COMPLETE pod=$POD rid=$RID api=$API_IP:$API_PORT ssh=$SSH_IP:$SSH_PORT terminate=$TERM_AT"
