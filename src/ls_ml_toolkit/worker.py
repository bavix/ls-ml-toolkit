#!/usr/bin/env python3
"""A training worker that polls S3 for jobs, instead of a message broker or a database.

This project already has an S3 bucket and boto3 credentials wired up (`config_loader.get_s3_config`)
for pulling dataset images. A queue is three prefixes in that same bucket:

    jobs/pending/<id>.json    -> submit_job() writes here
    jobs/running/<id>.json    -> the worker claims a job by moving it here (S3 has no atomic
                                  claim, so a crashed worker can leave a job stuck; that is an
                                  acceptable failure mode for one operator running one worker,
                                  and cheaper than a broker for that scale)
    jobs/done/<id>.json       -> the worker writes the result here when finished, alongside
                                  jobs/done/<id>/ holding the trained artifacts

No server, no open port, nothing to expose on a home machine — the worker only ever makes outbound
requests to S3. Submitting a job is `lsml-worker submit`; running the worker is `lsml-worker run`,
meant to sit in a terminal (or a service unit) on whatever machine is available, polling on an
interval. One worker at a time is the whole design: this exists because a machine is offered
occasionally, not because there is a fleet to schedule across. If that changes, replace this file —
the job JSON shape is the only contract anything else needs to keep.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 30


def _s3_client(config):
    import boto3

    kwargs = {
        "aws_access_key_id": config["access_key_id"],
        "aws_secret_access_key": config["secret_access_key"],
        "region_name": config.get("region") or "us-east-1",
    }
    if config.get("endpoint"):
        kwargs["endpoint_url"] = config["endpoint"]
    return boto3.client("s3", **kwargs)


def _load_config(config_path: Optional[str]):
    from .config_loader import load_config

    return load_config(config_path or "ls-ml-toolkit.yaml")


@dataclasses.dataclass
class Job:
    """One training request. Everything a worker needs, nothing it has to guess.

    `recipe` is passed straight through to `ultralytics.YOLO(...).train(**recipe)` — no allowlist
    of parameters, because the worker's job is to run training somewhere else, not to re-validate
    what the caller already decided. The caller is this same toolkit's own code, run by the same
    operator; a worker on a home machine has no untrusted submitters to defend against.
    """

    job_id: str
    dataset_s3_prefix: str
    base_model: str
    recipe: dict
    submitted_at: str
    submitted_by: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_json(raw: str) -> "Job":
        return Job(**json.loads(raw))


def submit_job(
    dataset_dir: str,
    base_model: str,
    recipe: dict,
    bucket: str,
    config_path: Optional[str] = None,
    prefix: str = "jobs",
) -> str:
    """Uploads a local dataset directory to S3 and writes the job spec. Returns the job id.

    The dataset travels with the job rather than being referenced by a path the worker might not
    have — a worker on a different machine has no access to this machine's filesystem, and a
    dataset built by `lsml-tagset`/`lsml-crops` is exactly the kind of thing that only exists
    locally until this uploads it.
    """
    config = _load_config(config_path)
    s3 = _s3_client(config.get_s3_config())
    job_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    dataset_prefix = f"{prefix}/data/{job_id}"

    local_root = Path(dataset_dir)
    files = [p for p in local_root.rglob("*") if p.is_file()]
    logger.info("Uploading %d files from %s to s3://%s/%s", len(files), local_root, bucket, dataset_prefix)
    for path in files:
        key = f"{dataset_prefix}/{path.relative_to(local_root)}"
        s3.upload_file(str(path), bucket, key)

    job = Job(
        job_id=job_id,
        dataset_s3_prefix=f"s3://{bucket}/{dataset_prefix}",
        base_model=base_model,
        recipe=recipe,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        submitted_by=socket.gethostname(),
    )
    s3.put_object(Bucket=bucket, Key=f"{prefix}/pending/{job_id}.json", Body=job.to_json())
    logger.info("Submitted job %s", job_id)
    return job_id


def _download_prefix(s3, bucket: str, s3_prefix: str, dest: Path) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(s3_prefix):].lstrip("/")
            if not relative:
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(target))


def _run_training(job: Job, dataset_dir: Path, work_dir: Path) -> Path:
    """Runs training in a subprocess so a crashed training run cannot take the worker down with
    it — the worker's own job-lifecycle bookkeeping (claim/finish) must survive a bad recipe or an
    OOM in the training process.
    """
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"No data.yaml in uploaded dataset at {dataset_dir}")

    script = work_dir / "_train.py"
    script.write_text(
        "from ultralytics import YOLO\n"
        f"m = YOLO({job.base_model!r})\n"
        "m.train(**{\n"
        f"    'data': {str(data_yaml)!r},\n"
        f"    'project': {str(work_dir / 'runs')!r},\n"
        "    'name': 'run', 'exist_ok': True,\n"
        f"    **{job.recipe!r},\n"
        "})\n"
        "m.export(format='onnx', opset=13, imgsz=" + str(job.recipe.get("imgsz", 640)) + ", simplify=False)\n"
    )
    log_path = work_dir / "train.log"
    with log_path.open("w") as log_file:
        subprocess.run(
            [sys.executable, str(script)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    weights = work_dir / "runs" / "run" / "weights"
    onnx_candidates = list(weights.glob("*.onnx"))
    if not onnx_candidates:
        raise RuntimeError(f"Training finished but produced no .onnx; see {log_path}")
    return weights


def _process_one(s3, bucket: str, prefix: str, key: str) -> None:
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    job = Job.from_json(raw)
    running_key = f"{prefix}/running/{job.job_id}.json"
    s3.put_object(Bucket=bucket, Key=running_key, Body=raw)
    s3.delete_object(Bucket=bucket, Key=key)
    logger.info("Claimed job %s", job.job_id)

    with tempfile.TemporaryDirectory(prefix=f"lsml-job-{job.job_id}-") as tmp:
        work_dir = Path(tmp)
        dataset_dir = work_dir / "dataset"
        dataset_dir.mkdir()
        assert job.dataset_s3_prefix.startswith(f"s3://{bucket}/")
        s3_prefix = job.dataset_s3_prefix[len(f"s3://{bucket}/"):]
        _download_prefix(s3, bucket, s3_prefix, dataset_dir)

        result = {"job_id": job.job_id, "finished_at": datetime.now(timezone.utc).isoformat()}
        try:
            weights_dir = _run_training(job, dataset_dir, work_dir)
            for artifact in weights_dir.glob("*"):
                s3.upload_file(
                    str(artifact), bucket, f"{prefix}/done/{job.job_id}/weights/{artifact.name}",
                )
            s3.upload_file(
                str(work_dir / "train.log"), bucket, f"{prefix}/done/{job.job_id}/train.log",
            )
            result["status"] = "success"
        except Exception as error:  # noqa: BLE001 - reported to the job record, not swallowed
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            result["traceback"] = traceback.format_exc()
            logger.exception("Job %s failed", job.job_id)

        s3.put_object(
            Bucket=bucket, Key=f"{prefix}/done/{job.job_id}.json",
            Body=json.dumps(result, indent=2),
        )
        s3.delete_object(Bucket=bucket, Key=running_key)
    logger.info("Finished job %s: %s", job.job_id, result["status"])


def run_worker(
    bucket: str,
    config_path: Optional[str] = None,
    prefix: str = "jobs",
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    once: bool = False,
) -> None:
    """Polls `jobs/pending/` and runs one job at a time, in submission order.

    One at a time on purpose: this runs on a machine offered occasionally, not a cluster, and
    running two training jobs at once on the same CPU would just make both slower without
    changing total throughput.
    """
    config = _load_config(config_path)
    s3 = _s3_client(config.get_s3_config())
    logger.info("Worker started on %s, polling s3://%s/%s/pending every %ds",
                socket.gethostname(), bucket, prefix, poll_seconds)

    while True:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/pending/")
        pending = sorted(obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json"))
        if pending:
            _process_one(s3, bucket, prefix, pending[0])
        elif once:
            logger.info("No pending jobs; --once set, exiting")
            return
        else:
            time.sleep(poll_seconds)
        if once and not pending:
            return


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="S3-queued training jobs — no broker, no server")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Upload a dataset and enqueue a training job")
    submit.add_argument("dataset_dir")
    submit.add_argument("--bucket", required=True)
    submit.add_argument("--base-model", default="yolov8n.pt")
    submit.add_argument("--epochs", type=int, default=30)
    submit.add_argument("--patience", type=int, default=10)
    submit.add_argument("--freeze", type=int, default=None)
    submit.add_argument("--imgsz", type=int, default=640)
    submit.add_argument("--batch", type=int, default=16)
    submit.add_argument("--device", default="cpu", help="The worker's own device; a home CPU box wants 'cpu'")
    submit.add_argument("--config", dest="config_path", default=None)
    submit.add_argument("--prefix", default="jobs")

    run = sub.add_parser("run", help="Poll for jobs and train them, one at a time")
    run.add_argument("--bucket", required=True)
    run.add_argument("--config", dest="config_path", default=None)
    run.add_argument("--prefix", default="jobs")
    run.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    run.add_argument("--once", action="store_true", help="Process at most one pending job, then exit")

    status = sub.add_parser("status", help="List pending/running/done jobs")
    status.add_argument("--bucket", required=True)
    status.add_argument("--config", dest="config_path", default=None)
    status.add_argument("--prefix", default="jobs")

    args = parser.parse_args()

    if args.command == "submit":
        recipe = {
            "epochs": args.epochs, "patience": args.patience,
            "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
        }
        if args.freeze is not None:
            recipe["freeze"] = args.freeze
        job_id = submit_job(
            args.dataset_dir, args.base_model, recipe, args.bucket, args.config_path, args.prefix,
        )
        print(job_id)
    elif args.command == "run":
        run_worker(args.bucket, args.config_path, args.prefix, args.poll_seconds, args.once)
    elif args.command == "status":
        config = _load_config(args.config_path)
        s3 = _s3_client(config.get_s3_config())
        for state in ("pending", "running", "done"):
            response = s3.list_objects_v2(Bucket=args.bucket, Prefix=f"{args.prefix}/{state}/")
            keys = [
                o["Key"].rsplit("/", 1)[-1].removesuffix(".json")
                for o in response.get("Contents", []) if o["Key"].endswith(".json")
            ]
            print(f"{state}: {len(keys)}")
            for k in keys:
                print(f"  {k}")


if __name__ == "__main__":
    main()
