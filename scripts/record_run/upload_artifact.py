"""Upload a results tarball to W&B as an artifact (durable off-pod sink)."""
import os
import wandb

rid = os.environ["RID"]
tag = os.environ["TAG"]
path = os.environ["TARPATH"]

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "pacevolve-rfg"),
    entity=os.environ.get("WANDB_ENTITY") or None,
    name=f"rfg-{rid}-artifact-{tag}",
    job_type="artifact-upload",
    settings=wandb.Settings(silent=True),
)
art = wandb.Artifact(f"rfg-{rid}-results", type="run-results", metadata={"tag": tag})
art.add_file(path)
run.log_artifact(art)
run.finish()
print(f"uploaded {path} as rfg-{rid}-results ({tag})")
