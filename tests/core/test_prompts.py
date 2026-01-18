"""Tests for backend/core/prompts.py."""

from backend.core.prompts import PromptService, prompts

# -----------------------------------------------------------------------------
# PromptService Singleton Tests
# -----------------------------------------------------------------------------


class TestPromptServiceSingleton:
    """Tests for the prompts singleton."""

    def test_prompts_is_instance(self) -> None:
        """Test that prompts is a PromptService instance."""
        assert isinstance(prompts, PromptService)


# -----------------------------------------------------------------------------
# conversation_tree_generator Tests
# -----------------------------------------------------------------------------


class TestConversationTreeGenerator:
    """Tests for conversation_tree_generator prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        system, user = prompts.conversation_tree_generator(
            num_nodes=5,
            conversation_goal="Help user debug code",
            conversation_context="My Python code isn't working",
        )

        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0

    def test_system_prompt_contains_role(self) -> None:
        """Test that system prompt defines the role."""
        system, _ = prompts.conversation_tree_generator(
            num_nodes=3,
            conversation_goal="Test",
            conversation_context="Test",
        )

        assert "strategic" in system.lower()
        assert "JSON" in system

    def test_user_prompt_contains_goal(self) -> None:
        """Test that user prompt includes the goal."""
        _, user = prompts.conversation_tree_generator(
            num_nodes=3,
            conversation_goal="Help with async programming",
            conversation_context="I need help",
        )

        assert "Help with async programming" in user
        assert "3" in user  # num_nodes

    def test_includes_research_context(self) -> None:
        """Test that research context is included when provided."""
        _, user = prompts.conversation_tree_generator(
            num_nodes=3,
            conversation_goal="Test",
            conversation_context="Test",
            deep_research_context="Async/await is important for I/O bound tasks...",
        )

        assert "Async/await" in user
        assert "Research context" in user

    def test_no_research_context(self) -> None:
        """Test that no research section when context is None."""
        _, user = prompts.conversation_tree_generator(
            num_nodes=3,
            conversation_goal="Test",
            conversation_context="Test",
            deep_research_context=None,
        )

        assert "Research context" not in user


# -----------------------------------------------------------------------------
# user_intent_generator Tests
# -----------------------------------------------------------------------------


class TestUserIntentGenerator:
    """Tests for user_intent_generator prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        system, user = prompts.user_intent_generator(
            num_intents=3,
            conversation_goal="Help user",
            conversation_history="User: Hello\nAssistant: Hi!",
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_defines_role(self) -> None:
        """Test that system prompt defines the analyzer role."""
        system, _ = prompts.user_intent_generator(
            num_intents=3,
            conversation_goal="Test",
            conversation_history="Test",
        )

        assert "analyze" in system.lower()
        assert "JSON" in system

    def test_user_prompt_contains_parameters(self) -> None:
        """Test that user prompt includes all parameters."""
        _, user = prompts.user_intent_generator(
            num_intents=5,
            conversation_goal="Debug Python code",
            conversation_history="User: My code broke",
        )

        assert "5" in user
        assert "Debug Python code" in user
        assert "My code broke" in user

    def test_includes_emotional_tones(self) -> None:
        """Test that emotional tones are listed."""
        _, user = prompts.user_intent_generator(
            num_intents=3,
            conversation_goal="Test",
            conversation_history="Test",
        )

        assert "engaged" in user.lower()
        assert "skeptical" in user.lower()


# -----------------------------------------------------------------------------
# user_simulation Tests
# -----------------------------------------------------------------------------


class TestUserSimulation:
    """Tests for user_simulation prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        system, user = prompts.user_simulation(
            conversation_goal="Help with debugging",
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_contains_goal(self) -> None:
        """Test that system prompt includes the goal."""
        system, _ = prompts.user_simulation(
            conversation_goal="Help with async programming",
        )

        assert "async programming" in system

    def test_includes_intent_when_provided(self) -> None:
        """Test that intent details are included."""
        system, _ = prompts.user_simulation(
            conversation_goal="Test",
            user_intent={
                "label": "Confused User",
                "description": "User doesn't understand",
                "emotional_tone": "confused",
                "cognitive_stance": "questioning",
            },
        )

        assert "Confused User" in system
        assert "questioning" in system

    def test_no_intent_section_when_none(self) -> None:
        """Test that no intent section when None."""
        system, _ = prompts.user_simulation(
            conversation_goal="Test",
            user_intent=None,
        )

        assert "MUST embody" not in system

    def test_user_prompt_is_continuation(self) -> None:
        """Test that user prompt requests continuation."""
        _, user = prompts.user_simulation(conversation_goal="Test")

        assert "Continue" in user or "continue" in user


# -----------------------------------------------------------------------------
# assistant_continuation Tests
# -----------------------------------------------------------------------------


class TestAssistantContinuation:
    """Tests for assistant_continuation prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        system, user = prompts.assistant_continuation(
            conversation_goal="Help user",
            strategy_tagline="Be Empathetic",
            strategy_description="Focus on understanding emotions",
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_contains_strategy(self) -> None:
        """Test that system prompt includes strategy."""
        system, _ = prompts.assistant_continuation(
            conversation_goal="Debug code",
            strategy_tagline="Step by Step",
            strategy_description="Walk through each step",
        )

        assert "Step by Step" in system
        assert "Walk through" in system
        assert "Debug code" in system


# -----------------------------------------------------------------------------
# rephrase_with_intent Tests
# -----------------------------------------------------------------------------


class TestRephraseWithIntent:
    """Tests for rephrase_with_intent prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        system, user = prompts.rephrase_with_intent(
            original_message="Help me with my code",
            intent_label="Frustrated",
            intent_description="User is annoyed",
            emotional_tone="frustrated",
            cognitive_stance="demanding",
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_user_prompt_contains_all_params(self) -> None:
        """Test that user prompt includes all parameters."""
        _, user = prompts.rephrase_with_intent(
            original_message="I need help with Python",
            intent_label="Curious",
            intent_description="Wants to learn more",
            emotional_tone="enthusiastic",
            cognitive_stance="exploring",
        )

        assert "I need help with Python" in user
        assert "Curious" in user
        assert "enthusiastic" in user
        assert "exploring" in user


# -----------------------------------------------------------------------------
# trajectory_outcome_judge Tests
# -----------------------------------------------------------------------------


class TestTrajectoryOutcomeJudge:
    """Tests for trajectory_outcome_judge prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        system, user = prompts.trajectory_outcome_judge(
            conversation_goal="Help user debug",
            conversation_history="User: Hi\nAssistant: Hello",
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_is_critical(self) -> None:
        """Test that system prompt is set up for critical evaluation."""
        system, _ = prompts.trajectory_outcome_judge(
            conversation_goal="Test",
            conversation_history="Test",
        )

        assert "EXACTING" in system or "HARSHLY" in system
        assert "JSON" in system

    def test_includes_criteria(self) -> None:
        """Test that evaluation criteria are listed."""
        _, user = prompts.trajectory_outcome_judge(
            conversation_goal="Test",
            conversation_history="Test",
        )

        assert "goal_achieved" in user
        assert "user_engagement" in user or "forward_progress" in user

    def test_includes_research_context(self) -> None:
        """Test that research context is included."""
        _, user = prompts.trajectory_outcome_judge(
            conversation_goal="Test",
            conversation_history="Test",
            deep_research_context="Research findings here...",
        )

        assert "Research findings here" in user


# -----------------------------------------------------------------------------
# branch_selection_judge Tests
# -----------------------------------------------------------------------------


class TestBranchSelectionJudge:
    """Tests for branch_selection_judge prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        system, user = prompts.branch_selection_judge(
            conversation_goal="Help user",
            conversation_context="Initial message",
            branch_tagline="Empathetic Approach",
            branch_description="Focus on emotions",
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_user_prompt_contains_branch_info(self) -> None:
        """Test that user prompt includes branch details."""
        _, user = prompts.branch_selection_judge(
            conversation_goal="Debug code",
            conversation_context="My code is broken",
            branch_tagline="Technical Deep Dive",
            branch_description="Analyze code in detail",
        )

        assert "Technical Deep Dive" in user
        assert "Analyze code in detail" in user
        assert "Debug code" in user


# -----------------------------------------------------------------------------
# comparative_trajectory_judge Tests
# -----------------------------------------------------------------------------


class TestComparativeTrajectoryJudge:
    """Tests for comparative_trajectory_judge prompt."""

    def test_returns_prompt_pair(self) -> None:
        """Test that method returns a tuple of two strings."""
        trajectories = [
            {"id": "t1", "intent_label": "Curious", "history": "User: Hi\nAssistant: Hello"},
            {"id": "t2", "intent_label": "Skeptical", "history": "User: Hi\nAssistant: Hey"},
        ]

        system, user = prompts.comparative_trajectory_judge(
            conversation_goal="Help user",
            trajectories=trajectories,
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_forces_ranking(self) -> None:
        """Test that system prompt emphasizes ranking."""
        system, _ = prompts.comparative_trajectory_judge(
            conversation_goal="Test",
            trajectories=[],
        )

        assert "rank" in system.lower() or "compare" in system.lower()

    def test_user_prompt_includes_trajectories(self) -> None:
        """Test that all trajectories are included."""
        trajectories = [
            {"id": "traj-1", "intent_label": "Eager", "history": "Conversation 1"},
            {"id": "traj-2", "intent_label": "Reluctant", "history": "Conversation 2"},
        ]

        _, user = prompts.comparative_trajectory_judge(
            conversation_goal="Test",
            trajectories=trajectories,
        )

        assert "traj-1" in user
        assert "traj-2" in user
        assert "Conversation 1" in user
        assert "Conversation 2" in user

    def test_includes_research_context(self) -> None:
        """Test that research context is included."""
        _, user = prompts.comparative_trajectory_judge(
            conversation_goal="Test",
            trajectories=[],
            deep_research_context="Important research...",
        )

        assert "Important research" in user
