"""
OpenEnv 0.2.1 wrapper for the Conversation Environment.

Provides OpenEnv-compatible registration and standardized interface.
If the openenv package is installed, this registers the environment.
Otherwise, it provides a standalone wrapper with the same API contract.
"""

from __future__ import annotations

from layer2.environment import ConversationEnvironment, EnvConfig, StepResult
from layer2.customer_sim import CustomerPersona, CustomerSimulator
from layer0.reward import BANKING_INTENTS

# Environment metadata for OpenEnv registry
ENV_ID = "nested-rl/CustomerSupport-v0"
ENV_METADATA = {
    "id": ENV_ID,
    "description": (
        "Multi-turn customer support conversation environment. "
        "An agent must classify customer intent while resisting social engineering."
    ),
    "action_space": "text",
    "observation_space": {
        "customer_message": "str",
        "domain": "str",
        "intents": "list[str]",
        "turn": "int",
    },
    "reward_range": (-150.0, 130.0),
    "max_episode_steps": 10,
    "domain": "banking",
    "intents": BANKING_INTENTS,
}


class OpenEnvCustomerSupport:
    """
    OpenEnv 0.2.1 compatible environment wrapper.

    Wraps ConversationEnvironment with the standardized OpenEnv interface:
    - reset() -> observation
    - step(action) -> (observation, reward, terminated, truncated, info)
    - metadata property
    """

    metadata = ENV_METADATA

    def __init__(
        self,
        personas: list[CustomerPersona] | None = None,
        simulator: CustomerSimulator | None = None,
        config: EnvConfig | None = None,
        persona_count: int = 100,
    ):
        if personas is None:
            from personas.generate_personas import generate_personas
            personas_data = generate_personas(persona_count)
            personas = [CustomerPersona(**p) for p in personas_data]

        self._simulator = simulator or CustomerSimulator()
        self._env = ConversationEnvironment(
            personas=personas,
            simulator=self._simulator,
            config=config or EnvConfig(),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed (for reproducibility)
            options: Optional dict with "persona_id" to select specific persona

        Returns:
            (observation, info)
        """
        import random
        if seed is not None:
            random.seed(seed)

        persona = None
        if options and "persona_id" in options:
            pid = options["persona_id"]
            if 0 <= pid < len(self._env.personas):
                persona = self._env.personas[pid]

        obs = self._env.reset(persona=persona)
        info = {
            "persona_id": self._env._current_persona.id,
            "social_engineering": self._env._current_persona.social_engineering,
            "complexity": self._env._current_persona.complexity,
        }
        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Agent's text response

        Returns:
            (observation, reward, terminated, truncated, info)
            - terminated: episode ended due to classification or injection
            - truncated: episode ended due to max turns
        """
        result = self._env.step(action)

        terminated = False
        truncated = False
        if result.done:
            reason = result.info.get("termination_reason", "")
            if reason == "max_turns_exceeded":
                truncated = True
            else:
                terminated = True

        return result.observation, result.reward, terminated, truncated, result.info

    def close(self):
        """Clean up resources."""
        pass

    def render(self) -> str:
        """Render the current conversation as text."""
        if not self._env._messages:
            return "(no conversation in progress)"

        lines = []
        for msg in self._env._messages:
            role = "Customer" if msg["role"] == "customer" else "Agent"
            lines.append(f"[{role}] {msg['content']}")
        return "\n".join(lines)


def make_env(**kwargs) -> OpenEnvCustomerSupport:
    """Factory function for creating the environment (OpenEnv compatible)."""
    return OpenEnvCustomerSupport(**kwargs)


# Register with OpenEnv if available
try:
    import openenv
    openenv.register(
        id=ENV_ID,
        entry_point="layer2.openenv_wrapper:make_env",
        kwargs={},
    )
except (ImportError, AttributeError):
    pass  # OpenEnv not installed; wrapper still works standalone
