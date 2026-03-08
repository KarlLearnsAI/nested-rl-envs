"""
Supabase uploader for training results — incremental mode.

Uploads after every training step so data is never lost if the job crashes.

- Creates a training_runs row at the start of training
- Upserts that row after each step with updated reward arrays
- Inserts per-episode rows after each step

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


class SupabaseUploader:
    """
    Incremental uploader — call after_step() after each training step.

    Creates the training_runs row on first call, then upserts it with
    updated arrays on every subsequent call. Episode rows are inserted
    immediately and never re-sent.
    """

    def __init__(
        self,
        run_id: str,
        bucket: str = "training-results",
        config: dict[str, Any] | None = None,
    ):
        self.run_id = run_id
        self.bucket = bucket
        self.config = config
        self._client = _get_client()
        self._run_created = False

        # Accumulated arrays (mirrors what training_runs stores)
        self._mean_rewards: list[float] = []
        self._min_rewards: list[float] = []
        self._max_rewards: list[float] = []
        self._total_episodes = 0
        self._started_at = datetime.now(timezone.utc).isoformat()

        if self._client:
            logger.info("SupabaseUploader ready: run_id=%s", run_id)
            self._write_init_row()
        else:
            logger.warning("SupabaseUploader: no client — uploads will be skipped")

    def _write_init_row(self):
        """Write an init row to verify DB connectivity at startup."""
        try:
            run_row = {
                "run_id": self.run_id,
                "started_at": self._started_at,
                "duration_seconds": None,
                "total_steps": 0,
                "total_episodes": 0,
                "best_step": 0,
                "best_mean_reward": 0.0,
                "mean_rewards": [],
                "min_rewards": [],
                "max_rewards": [],
                "config": self.config,
            }
            self._client.table("training_runs").upsert(
                run_row, on_conflict="run_id"
            ).execute()
            self._run_created = True
            logger.info("DB init row written successfully (run_id=%s)", self.run_id)
        except Exception as e:
            logger.error("DB init row FAILED — check connection: %s", e)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def after_step(self, step: int, eval_result: dict[str, Any], prompt: str):
        """
        Called after each training step/candidate evaluation.

        Upserts the training_runs row and inserts new episode rows.
        """
        if not self._client:
            return

        mean_reward = eval_result.get("mean_reward", 0.0)
        min_reward = eval_result.get("min_reward", 0.0)
        max_reward = eval_result.get("max_reward", 0.0)

        self._mean_rewards.append(mean_reward)
        self._min_rewards.append(min_reward)
        self._max_rewards.append(max_reward)

        num_episodes = eval_result.get("num_episodes", 0)
        self._total_episodes += num_episodes

        # Best so far
        best_mean = max(self._mean_rewards)
        best_idx = self._mean_rewards.index(best_mean)

        # --- Upsert training_runs row ---
        run_row = {
            "run_id": self.run_id,
            "started_at": self._started_at,
            "duration_seconds": None,  # updated at end
            "total_steps": len(self._mean_rewards),
            "total_episodes": self._total_episodes,
            "best_step": best_idx,
            "best_mean_reward": best_mean,
            "mean_rewards": self._mean_rewards,
            "min_rewards": self._min_rewards,
            "max_rewards": self._max_rewards,
            "config": self.config,
        }

        try:
            self._client.table("training_runs").upsert(
                run_row, on_conflict="run_id"
            ).execute()
            self._run_created = True
            logger.info(
                "Upserted training_runs: step=%d mean_reward=%.1f",
                step, mean_reward,
            )
        except Exception as e:
            logger.error("Failed to upsert training_runs: %s", e)

        # --- Insert episode rows for this step ---
        episode_rows = []
        rewards_list = eval_result.get("rewards", [])
        for ei, log in enumerate(eval_result.get("logs", [])):
            episode_rows.append({
                "run_id": self.run_id,
                "step": step,
                "episode": ei,
                "reward": rewards_list[ei] if ei < len(rewards_list) else None,
                "turns": log.get("turns", 0),
                "intent_captured": log.get("intent_captured", False),
                "intent_correct": log.get("intent_correct", False),
                "true_intent": log.get("true_intent", ""),
                "agent_intent": log.get("agent_intent", ""),
                "injection_attempted": log.get("injection_attempted", False),
                "injection_succeeded": log.get("injection_succeeded", False),
                "api_call_made": log.get("api_call_made", False),
                "api_call_correct": log.get("api_call_correct", False),
            })

        if episode_rows:
            try:
                self._client.table("training_episodes").insert(episode_rows).execute()
                logger.info(
                    "Inserted %d episode rows for step %d", len(episode_rows), step
                )
            except Exception as e:
                logger.error("Failed to insert episodes for step %d: %s", step, e)

    def finish(
        self,
        duration_seconds: float | None = None,
        report_path: str | None = None,
        chart_path: str | None = None,
        raw_summary: dict[str, Any] | None = None,
    ):
        """
        Called at end of training. Updates duration and uploads final files.
        """
        if not self._client:
            return

        # Update duration on the run row
        if duration_seconds is not None and self._run_created:
            try:
                self._client.table("training_runs").update(
                    {"duration_seconds": duration_seconds}
                ).eq("run_id", self.run_id).execute()
                logger.info("Updated duration: %.1fs", duration_seconds)
            except Exception as e:
                logger.error("Failed to update duration: %s", e)

        # Upload files to Storage
        if raw_summary:
            self._upload_file(
                f"{self.run_id}/raw_summary.json",
                json.dumps(raw_summary, indent=2, default=str).encode(),
                "application/json",
            )

        if report_path and os.path.exists(report_path):
            with open(report_path, "rb") as f:
                self._upload_file(
                    f"{self.run_id}/report.md", f.read(), "text/markdown"
                )

        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                self._upload_file(
                    f"{self.run_id}/reward_chart.png", f.read(), "image/png"
                )

    def _upload_file(self, path: str, data: bytes, content_type: str):
        """Upload a single file to Supabase Storage."""
        try:
            self._client.storage.from_(self.bucket).upload(
                path, data, {"content-type": content_type}
            )
            logger.info("Uploaded %s to storage", path)
        except Exception as e:
            logger.error("Failed to upload %s: %s", path, e)
