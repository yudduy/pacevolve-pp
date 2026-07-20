# Pod-autonomous RunPod record-run chain

As-run 2026-07-19 for RID 719800 (Qwen3-32B TP4 record attempt on rectangle_free_grid).
The pod needs no babysitting after ignition: it trains, uploads results to W&B, and kills
its own billing.

```
launch.sh 32b|8b <RID>          (Mac) create pod w/ terminate-after + capacity ladder,
                                 push payload (repo + ttt_discover pkg + 5-key .env subset,
                                 tar-over-ssh — secure images lack rsync), ignite:
  bootstrap_{32b_tp4,8b}.sh     (pod) pinned skyrl-tx + all measured OOM/CUDA-graph fixes -> server
  pod_chain.sh <RID> <HF> 4 128 (pod) driver venv -> wait server -> run_advisor_rl 128 steps
                                 -> W&B artifact rfg-<RID>-results every 2h + at end
  pod_reaper.sh <RID>           (pod) on CHAIN COMPLETE or dead chain: emergency-upload,
                                 then stop the container from inside (=> GPU billing ends;
                                 no runpodctl/self-remove on secure images)
run_monitor.sh                  (Mac, optional) step rewards, stalls, EXITED/GONE, hourly balance
```

State file `record_run.state` (written by launch.sh next to itself) carries pod id +
endpoints. Harvest: W&B artifact `rfg-<RID>-results` (project pacevolve-rfg), or
`/workspace/pp/tasks/rectangle_free_grid/results/job_<RID>` on the pod while it lives.
After harvest: `runpodctl remove pod <id>` to clear the stopped shell.
