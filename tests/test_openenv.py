"""Tests for OpenEnv wrapper."""

import os
import pytest

from layer2.openenv_wrapper import OpenEnvCustomerSupport, ENV_METADATA

requires_hf_token = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required for LLM-based tests",
)


class TestOpenEnvWrapper:
    def test_metadata(self):
        env = OpenEnvCustomerSupport()
        assert env.metadata["id"] == "nested-rl/CustomerSupport-v0"
        assert env.metadata["action_space"] == "text"

    def test_reset_returns_tuple(self):
        env = OpenEnvCustomerSupport()
        obs, info = env.reset(seed=42)
        assert "customer_message" in obs
        assert "persona_id" in info

    def test_step_returns_5_tuple(self):
        env = OpenEnvCustomerSupport()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step('{"intent": "transfer"}')
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    @requires_hf_token
    def test_render(self):
        env = OpenEnvCustomerSupport()
        env.reset(seed=42)
        env.step("Hello, how can I help?")
        rendered = env.render()
        assert "[Customer]" in rendered
        assert "[Agent]" in rendered

    def test_reset_with_persona_id(self):
        env = OpenEnvCustomerSupport()
        obs, info = env.reset(options={"persona_id": 5})
        # persona_id in info reflects the persona's .id field, not the list index
        assert "persona_id" in info
        assert isinstance(info["persona_id"], int)
