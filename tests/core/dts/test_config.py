"""Tests for backend/core/dts/config.py."""

from backend.core.dts.config import DTSConfig


class TestDTSConfig:
    """Tests for DTSConfig dataclass."""

    def test_config_required_fields(self) -> None:
        """Test that goal and first_message are required."""
        config = DTSConfig(
            goal="Help user debug code",
            first_message="I have a bug in my code",
        )
        assert config.goal == "Help user debug code"
        assert config.first_message == "I have a bug in my code"

    def test_config_default_values(self) -> None:
        """Test default configuration values."""
        config = DTSConfig(goal="test", first_message="test")

        assert config.init_branches == 6
        assert config.deep_research is False
        assert config.research_cache_dir == ".cache/research"
        assert config.turns_per_branch == 5
        assert config.user_intents_per_branch == 3
        assert config.user_variability is False
        assert config.scoring_mode == "comparative"
        assert config.prune_threshold == 6.5
        assert config.keep_top_k is None
        assert config.min_survivors == 1
        assert config.max_concurrency == 16
        assert config.model is None
        assert config.strategy_model is None
        assert config.simulator_model is None
        assert config.judge_model is None
        assert config.temperature == 0.7
        assert config.judge_temperature == 0.3
        assert config.reasoning_enabled is False
        assert config.provider is None

    def test_config_custom_values(self) -> None:
        """Test setting custom configuration values."""
        config = DTSConfig(
            goal="Custom goal",
            first_message="Custom message",
            init_branches=10,
            deep_research=True,
            turns_per_branch=3,
            user_intents_per_branch=5,
            user_variability=True,
            scoring_mode="absolute",
            prune_threshold=7.0,
            keep_top_k=5,
            min_survivors=2,
            max_concurrency=8,
            model="gpt-4",
            strategy_model="gpt-4-turbo",
            simulator_model="gpt-3.5-turbo",
            judge_model="gpt-4",
            temperature=0.5,
            judge_temperature=0.2,
            reasoning_enabled=True,
            provider="Fireworks",
        )

        assert config.init_branches == 10
        assert config.deep_research is True
        assert config.turns_per_branch == 3
        assert config.user_intents_per_branch == 5
        assert config.user_variability is True
        assert config.scoring_mode == "absolute"
        assert config.prune_threshold == 7.0
        assert config.keep_top_k == 5
        assert config.min_survivors == 2
        assert config.max_concurrency == 8
        assert config.model == "gpt-4"
        assert config.strategy_model == "gpt-4-turbo"
        assert config.simulator_model == "gpt-3.5-turbo"
        assert config.judge_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.judge_temperature == 0.2
        assert config.reasoning_enabled is True
        assert config.provider == "Fireworks"

    def test_scoring_mode_absolute(self) -> None:
        """Test absolute scoring mode."""
        config = DTSConfig(
            goal="test",
            first_message="test",
            scoring_mode="absolute",
        )
        assert config.scoring_mode == "absolute"

    def test_scoring_mode_comparative(self) -> None:
        """Test comparative scoring mode."""
        config = DTSConfig(
            goal="test",
            first_message="test",
            scoring_mode="comparative",
        )
        assert config.scoring_mode == "comparative"

    def test_config_is_dataclass(self) -> None:
        """Test that DTSConfig is a dataclass."""
        import dataclasses

        assert dataclasses.is_dataclass(DTSConfig)

    def test_config_immutability(self) -> None:
        """Test that config values can be changed (dataclass is mutable by default)."""
        config = DTSConfig(goal="test", first_message="test")
        config.goal = "new goal"
        assert config.goal == "new goal"

    def test_config_with_per_phase_models(self) -> None:
        """Test configuration with different models per phase."""
        config = DTSConfig(
            goal="test",
            first_message="test",
            strategy_model="claude-3-opus",
            simulator_model="gpt-4-turbo",
            judge_model="claude-3-sonnet",
        )

        assert config.strategy_model == "claude-3-opus"
        assert config.simulator_model == "gpt-4-turbo"
        assert config.judge_model == "claude-3-sonnet"
        assert config.model is None  # Default model not set

    def test_config_research_settings(self) -> None:
        """Test research-related settings."""
        config = DTSConfig(
            goal="test",
            first_message="test",
            deep_research=True,
            research_cache_dir="/custom/cache",
        )

        assert config.deep_research is True
        assert config.research_cache_dir == "/custom/cache"
