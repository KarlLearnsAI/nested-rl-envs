"""
Supabase uploader for training results.

Uploads:
  1. Raw summary JSON + report files to Supabase Storage
  2. Per-run and per-episode metrics to Postgres tables

Requires SUPABASE_URL and SUPABASE_KEY environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _get_client():
    """Create a Supabase client from environment variables."""
    try:
        from supabase import create_client
    except ImportError:
        logger.error(
            "supabase package not installed. Install with: pip install 'nested-rl-envs[upload]'"
        )
        return None

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set")
        return None

    return create_client(url, key)


def upload_training_results(
    raw_summary: dict[str, Any],
    run_id: str | None = None,
    bucket: str = "training-results",
    report_path: str | None = None,
    chart_path: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Upload training results to Supabase (Storage + DB).

    Args:
        raw_summary: Output of TrainingLogger.generate_raw_summary().
        run_id: Unique run identifier. Auto-generated if not provided.
        bucket: Supabase Storage bucket name.
        report_path: Path to the markdown report file (optional).
        chart_path: Path to the reward chart PNG (optional).
        config: Training config dict to store with the run (optional).

    Returns:
        Dict with upload results: {"run_id", "storage_paths", "db_rows"}.
    """
    client = _get_client()
    if client is None:
        logger.warning("Supabase upload skipped — client not available")
        return {"run_id": None, "storage_paths": [], "db_rows": 0, "error": "no client"}

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results: dict[str, Any] = {"run_id": run_id, "storage_paths": [], "db_rows": 0}

    # --- Storage uploads ---
    results["storage_paths"] = _upload_files(
        client, bucket, run_id, raw_summary, report_path, chart_path
    )

    # --- DB inserts ---
    results["db_rows"] = _insert_metrics(client, run_id, raw_summary, config)

    logger.info(
        "Supabase upload complete: run_id=%s, files=%d, db_rows=%d",
        run_id, len(results["storage_paths"]), results["db_rows"],
    )
    return results


def _upload_files(
    client,
    bucket: str,
    run_id: str,
    raw_summary: dict[str, Any],
    report_path: str | None,
    chart_path: str | None,
) -> list[str]:
    """Upload files to Supabase Storage."""
    uploaded = []

    # Upload raw summary JSON
    try:
        summary_bytes = json.dumps(raw_summary, indent=2, default=str).encode()
        path = f"{run_id}/raw_summary.json"
        client.storage.from_(bucket).upload(
            path, summary_bytes, {"content-type": "application/json"}
        )
        uploaded.append(path)
        logger.info("Uploaded %s to storage", path)
    except Exception as e:
        logger.error("Failed to upload raw_summary.json: %s", e)

    # Upload report markdown
    if report_path and os.path.exists(report_path):
        try:
            with open(report_path, "rb") as f:
                path = f"{run_id}/report.md"
                client.storage.from_(bucket).upload(
                    path, f.read(), {"content-type": "text/markdown"}
                )
                uploaded.append(path)
                logger.info("Uploaded %s to storage", path)
        except Exception as e:
            logger.error("Failed to upload report: %s", e)

    # Upload chart PNG
    if chart_path and os.path.exists(chart_path):
        try:
            with open(chart_path, "rb") as f:
                path = f"{run_id}/reward_chart.png"
                client.storage.from_(bucket).upload(
                    path, f.read(), {"content-type": "image/png"}
                )
                uploaded.append(path)
                logger.info("Uploaded %s to storage", path)
        except Exception as e:
            logger.error("Failed to upload chart: %s", e)

    return uploaded


def _insert_metrics(
    client,
    run_id: str,
    raw_summary: dict[str, Any],
    config: dict[str, Any] | None,
) -> int:
    """Insert training run + per-episode metrics into Postgres tables."""
    rows_inserted = 0

    # Insert training run summary
    try:
        run_row = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": raw_summary.get("duration_seconds"),
            "total_steps": len(raw_summary.get("steps", [])),
            "total_episodes": raw_summary.get("total_episodes", 0),
            "best_step": raw_summary.get("best_step"),
            "best_mean_reward": raw_summary.get("best_mean_reward"),
            "mean_rewards": raw_summary.get("mean_rewards", []),
            "min_rewards": raw_summary.get("min_rewards", []),
            "max_rewards": raw_summary.get("max_rewards", []),
            "config": config,
        }
        client.table("training_runs").insert(run_row).execute()
        rows_inserted += 1
        logger.info("Inserted training run: %s", run_id)
    except Exception as e:
        logger.error("Failed to insert training_runs row: %s", e)

    # Insert per-episode metrics in batches
    episode_rows = []
    for m in raw_summary.get("per_episode_metrics", []):
        episode_rows.append({
            "run_id": run_id,
            "step": m["step"],
            "episode": m["episode"],
            "reward": m.get("reward"),
            "turns": m.get("turns", 0),
            "intent_captured": m.get("intent_captured", False),
            "intent_correct": m.get("intent_correct", False),
            "true_intent": m.get("true_intent", ""),
            "agent_intent": m.get("agent_intent", ""),
            "injection_attempted": m.get("injection_attempted", False),
            "injection_succeeded": m.get("injection_succeeded", False),
            "api_call_made": m.get("api_call_made", False),
            "api_call_correct": m.get("api_call_correct", False),
        })

    # Batch insert (Supabase/PostgREST supports bulk inserts)
    if episode_rows:
        batch_size = 100
        for i in range(0, len(episode_rows), batch_size):
            batch = episode_rows[i : i + batch_size]
            try:
                client.table("training_episodes").insert(batch).execute()
                rows_inserted += len(batch)
            except Exception as e:
                logger.error("Failed to insert episode batch %d: %s", i, e)

    return rows_inserted
