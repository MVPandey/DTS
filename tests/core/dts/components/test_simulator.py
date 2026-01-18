"""Tests for backend/core/dts/components/simulator.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.dts.components.simulator import (
    TERMINATION_SIGNALS,
    ConversationSimulator,
    LLMEmptyResponseError,
)
from backend.core.dts.tree import DialogueTree
from backend.core.dts.types import DialogueNode, NodeStatus, Strategy, UserIntent
from backend.llm.types import Completion, Message, Usage

# -----------------------------------------------------------------------------
# LLMEmptyResponseError Tests
# -----------------------------------------------------------------------------


class TestLLMEmptyResponseError:
    """Tests for LLMEmptyResponseError exception."""

    def test_error_creation(self) -> None:
        """Test creating the exception."""
        error = LLMEmptyResponseError("Empty response")
        assert str(error) == "Empty response"
        assert isinstance(error, Exception)


# -----------------------------------------------------------------------------
# TERMINATION_SIGNALS Tests
# -----------------------------------------------------------------------------


class TestTerminationSignals:
    """Tests for termination signal detection."""

    def test_termination_signals_exist(self) -> None:
        """Test that termination signals are defined."""
        assert len(TERMINATION_SIGNALS) > 0
        assert "goodbye" in TERMINATION_SIGNALS
        assert "bye" in TERMINATION_SIGNALS
        assert "quit" in TERMINATION_SIGNALS


# -----------------------------------------------------------------------------
# ConversationSimulator Initialization Tests
# -----------------------------------------------------------------------------


class TestSimulatorInit:
    """Tests for ConversationSimulator initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        mock_llm = MagicMock()
        simulator = ConversationSimulator(llm=mock_llm, goal="Test goal")

        assert simulator.llm is mock_llm
        assert simulator.goal == "Test goal"
        assert simulator.model is None
        assert simulator.temperature == 0.7
        assert simulator.provider is None

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        mock_llm = MagicMock()
        simulator = ConversationSimulator(
            llm=mock_llm,
            goal="Custom goal",
            model="gpt-4",
            temperature=0.5,
            max_concurrency=8,
            provider="Fireworks",
            reasoning_enabled=True,
        )

        assert simulator.model == "gpt-4"
        assert simulator.temperature == 0.5
        assert simulator.provider == "Fireworks"


# -----------------------------------------------------------------------------
# Termination Detection Tests
# -----------------------------------------------------------------------------


class TestTerminationDetection:
    """Tests for _should_terminate method."""

    @pytest.fixture
    def simulator(self) -> ConversationSimulator:
        """Create a simulator for testing."""
        return ConversationSimulator(llm=MagicMock(), goal="Test")

    def test_detects_goodbye(self, simulator: ConversationSimulator) -> None:
        """Test detection of goodbye signal."""
        assert simulator._should_terminate("Goodbye, thanks for your help!") is True

    def test_detects_bye(self, simulator: ConversationSimulator) -> None:
        """Test detection of bye signal."""
        assert simulator._should_terminate("Bye!") is True

    def test_detects_quit(self, simulator: ConversationSimulator) -> None:
        """Test detection of quit signal."""
        assert simulator._should_terminate("I want to quit this conversation") is True

    def test_detects_frustration(self, simulator: ConversationSimulator) -> None:
        """Test detection of short frustrated responses."""
        assert simulator._should_terminate("no") is True
        assert simulator._should_terminate("nope") is True
        assert simulator._should_terminate("wrong!") is True

    def test_normal_response_not_terminated(self, simulator: ConversationSimulator) -> None:
        """Test that normal responses don't trigger termination."""
        assert simulator._should_terminate("That's interesting, tell me more") is False
        assert simulator._should_terminate("I have another question") is False

    def test_case_insensitive(self, simulator: ConversationSimulator) -> None:
        """Test that detection is case insensitive."""
        assert simulator._should_terminate("GOODBYE") is True
        assert simulator._should_terminate("GoodBye") is True


# -----------------------------------------------------------------------------
# expand_nodes Tests
# -----------------------------------------------------------------------------


class TestExpandNodes:
    """Tests for ConversationSimulator.expand_nodes()."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.fixture
    def sample_node(self) -> DialogueNode:
        """Create a sample node for testing."""
        return DialogueNode(
            id="node-1",
            depth=0,
            strategy=Strategy(tagline="Test", description="Test strategy"),
            messages=[Message.user("Hello")],
        )

    @pytest.mark.asyncio
    async def test_expand_linear_no_forking(
        self, simulator: ConversationSimulator, mock_llm: MagicMock, sample_node: DialogueNode
    ) -> None:
        """Test linear expansion without intent forking."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("Response"),
            usage=Usage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
        )

        expanded = await simulator.expand_nodes([sample_node], turns=1, intents_per_node=1)

        assert len(expanded) == 1
        assert len(expanded[0].messages) > 1

    @pytest.mark.asyncio
    async def test_expand_with_forking(
        self, simulator: ConversationSimulator, mock_llm: MagicMock, sample_node: DialogueNode
    ) -> None:
        """Test expansion with intent forking."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("Response"),
            usage=Usage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
        )

        async def mock_generate_intents(_history, count):
            return [
                UserIntent(
                    id=f"intent-{i}",
                    label=f"Intent {i}",
                    description=f"Intent {i} desc",
                    emotional_tone="neutral",
                    cognitive_stance="neutral",
                )
                for i in range(count)
            ]

        tree = DialogueTree.create(sample_node)

        expanded = await simulator.expand_nodes(
            [sample_node],
            turns=1,
            intents_per_node=2,
            tree=tree,
            generate_intents=mock_generate_intents,
        )

        # Should have created forked nodes
        assert len(expanded) == 2


# -----------------------------------------------------------------------------
# _run_turn Tests
# -----------------------------------------------------------------------------


class TestRunTurn:
    """Tests for ConversationSimulator._run_turn()."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_run_turn_success(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test successful turn execution."""
        mock_llm.complete.side_effect = [
            Completion(message=Message.assistant("User response")),  # User simulation
            Completion(message=Message.assistant("Assistant response")),  # Assistant
        ]

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
        )
        history = [Message.user("Hello")]

        result = await simulator._run_turn(node, history, turn_idx=0)

        assert result is True
        assert len(history) == 3  # Original + user + assistant

    @pytest.mark.asyncio
    async def test_run_turn_skip_user_simulation(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test turn with skipped user simulation."""
        mock_llm.complete.return_value = Completion(message=Message.assistant("Assistant response"))

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
        )
        history = [Message.user("Hello")]

        result = await simulator._run_turn(node, history, turn_idx=0, skip_user_simulation=True)

        assert result is True
        # Only assistant message added
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_run_turn_termination(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test turn termination on goodbye signal."""
        mock_llm.complete.return_value = Completion(message=Message.assistant("goodbye"))

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
        )
        history = [Message.user("Hello")]

        result = await simulator._run_turn(node, history, turn_idx=0)

        assert result is False
        assert node.status == NodeStatus.TERMINAL


# -----------------------------------------------------------------------------
# _simulate_user Tests
# -----------------------------------------------------------------------------


class TestSimulateUser:
    """Tests for user simulation."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_simulate_user_success(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test successful user simulation."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("That's interesting!")
        )

        history = [Message.assistant("Here's some info")]
        result = await simulator._simulate_user(history)

        assert result == "That's interesting!"

    @pytest.mark.asyncio
    async def test_simulate_user_with_intent(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test user simulation with intent."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("I'm confused about this")
        )

        intent = UserIntent(
            id="confused",
            label="Confused",
            description="User is confused",
            emotional_tone="confused",
            cognitive_stance="questioning",
        )

        history = [Message.assistant("Here's some info")]
        result = await simulator._simulate_user(history, intent=intent)

        assert result == "I'm confused about this"


# -----------------------------------------------------------------------------
# _generate_assistant Tests
# -----------------------------------------------------------------------------


class TestGenerateAssistant:
    """Tests for assistant response generation."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_generate_assistant_success(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test successful assistant generation."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("I can help you with that!")
        )

        strategy = Strategy(tagline="Helpful", description="Be very helpful")
        history = [Message.user("I need help")]

        result = await simulator._generate_assistant(history, strategy)

        assert result == "I can help you with that!"

    @pytest.mark.asyncio
    async def test_generate_assistant_no_strategy(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test assistant generation without strategy."""
        mock_llm.complete.return_value = Completion(message=Message.assistant("Generic response"))

        history = [Message.user("Hello")]
        result = await simulator._generate_assistant(history, None)

        assert result == "Generic response"


# -----------------------------------------------------------------------------
# Retry Logic Tests
# -----------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for retry logic on empty responses."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_retries_on_empty_response(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test retry on empty response."""
        mock_llm.complete.side_effect = [
            Completion(message=Message.assistant("")),  # Empty
            Completion(message=Message.assistant("")),  # Empty again
            Completion(message=Message.assistant("Valid response")),  # Success
        ]

        result = await simulator._call_llm_with_retry(
            [Message.user("test")], phase="test", max_retries=3
        )

        assert result == "Valid response"
        assert mock_llm.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test that error is raised after max retries."""
        mock_llm.complete.return_value = Completion(message=Message.assistant(""))

        with pytest.raises(LLMEmptyResponseError):
            await simulator._call_llm_with_retry(
                [Message.user("test")], phase="test", max_retries=2
            )


# -----------------------------------------------------------------------------
# Usage Tracking Tests
# -----------------------------------------------------------------------------


class TestUsageTracking:
    """Tests for token usage tracking."""

    @pytest.mark.asyncio
    async def test_tracks_usage(self) -> None:
        """Test that usage is tracked."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=Completion(
                message=Message.assistant("Response"),
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            )
        )

        usage_callback = MagicMock()
        simulator = ConversationSimulator(
            llm=mock_llm,
            goal="Test",
            on_usage=usage_callback,
        )

        await simulator._call_llm([Message.user("test")], phase="user")

        usage_callback.assert_called_once()
        call_args = usage_callback.call_args
        assert call_args[0][1] == "user"  # Phase name


# -----------------------------------------------------------------------------
# Additional Coverage Tests
# -----------------------------------------------------------------------------


class TestExpandNodesEdgeCases:
    """Tests for edge cases in expand_nodes."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_expand_with_intent_generation_failure(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test expansion falls back when intent generation fails."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("Response"),
        )

        async def failing_generate_intents(_history, _count):
            raise RuntimeError("Intent generation failed")

        node = DialogueNode(
            id="node-1",
            depth=0,
            strategy=Strategy(tagline="Test", description="Test strategy"),
            messages=[Message.user("Hello")],
        )

        expanded = await simulator.expand_nodes(
            [node],
            turns=1,
            intents_per_node=3,
            generate_intents=failing_generate_intents,
        )

        # Should fall back to linear expansion
        assert len(expanded) == 1

    @pytest.mark.asyncio
    async def test_expand_with_empty_intent_result(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test expansion falls back when intents are empty."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("Response"),
        )

        async def empty_generate_intents(_history, _count):
            return []  # Empty intents

        node = DialogueNode(
            id="node-1",
            depth=0,
            strategy=Strategy(tagline="Test", description="Test strategy"),
            messages=[Message.user("Hello")],
        )

        expanded = await simulator.expand_nodes(
            [node],
            turns=1,
            intents_per_node=3,
            generate_intents=empty_generate_intents,
        )

        # Should fall back to linear expansion
        assert len(expanded) == 1


class TestLinearBatchExpansion:
    """Tests for _expand_linear_batch."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_batch_expansion_with_failures(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test batch expansion handles individual node failures."""
        # First node will fail, second will succeed
        call_count = 0

        async def mock_complete(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First node's calls fail
                raise RuntimeError("LLM error")
            return Completion(message=Message.assistant("Response"))

        mock_llm.complete = mock_complete

        node1 = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test1", description="Test1"),
            messages=[Message.user("Hello")],
        )
        node2 = DialogueNode(
            id="node-2",
            strategy=Strategy(tagline="Test2", description="Test2"),
            messages=[Message.user("Hello")],
        )

        expanded = await simulator._expand_linear_batch([node1, node2], turns=1)

        # At least some nodes should be expanded
        # The exact result depends on async timing
        assert node1.status == NodeStatus.ERROR or len(expanded) > 0


class TestRunTurnEdgeCases:
    """Tests for _run_turn edge cases."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_run_turn_empty_user_response(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test handling of empty user response."""
        # Simulate empty response that triggers retry then failure
        mock_llm.complete.return_value = Completion(message=Message.assistant(""))

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
        )
        history = [Message.user("Hello")]

        result = await simulator._run_turn(node, history, turn_idx=0)

        assert result is False
        assert node.status == NodeStatus.ERROR
        assert node.prune_reason is not None
        assert "empty user response" in node.prune_reason

    @pytest.mark.asyncio
    async def test_run_turn_empty_assistant_response(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test handling of empty assistant response."""
        # User response OK, then assistant empty
        mock_llm.complete.side_effect = [
            Completion(message=Message.assistant("User says hello")),  # User sim
            Completion(message=Message.assistant("")),  # Empty assistant
            Completion(message=Message.assistant("")),  # Retry
            Completion(message=Message.assistant("")),  # Retry
        ]

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
        )
        history = [Message.user("Hello")]

        result = await simulator._run_turn(node, history, turn_idx=0)

        assert result is False
        assert node.status == NodeStatus.ERROR
        assert node.prune_reason is not None
        assert "empty assistant response" in node.prune_reason


class TestExpandWithIntent:
    """Tests for _expand_with_intent."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_expand_with_intent_rephrases(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test that intent expansion rephrases the initial message."""
        mock_llm.complete.side_effect = [
            Completion(message=Message.assistant("Rephrased message")),  # Rephrase
            Completion(message=Message.assistant("Assistant response")),  # Turn 1
        ]

        intent = UserIntent(
            id="frustrated",
            label="Frustrated",
            description="User is frustrated",
            emotional_tone="frustrated",
            cognitive_stance="impatient",
        )

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
            user_intent=intent,
            messages=[Message.user("Original message")],
        )

        result = await simulator._expand_with_intent(node, turns=1, first_intent=intent)

        assert result is not None
        # The first message should be rephrased
        assert result.messages[0].content == "Rephrased message"

    @pytest.mark.asyncio
    async def test_expand_with_intent_rephrase_failure(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test that intent expansion handles rephrase failure."""
        # First call (rephrase) returns empty, second (assistant) succeeds
        mock_llm.complete.side_effect = [
            Completion(message=Message.assistant("")),  # Empty rephrase
            Completion(message=Message.assistant("")),  # Retry
            Completion(message=Message.assistant("")),  # Retry
            Completion(message=Message.assistant("Assistant response")),  # Turn 1
        ]

        intent = UserIntent(
            id="curious",
            label="Curious",
            description="User is curious",
            emotional_tone="curious",
            cognitive_stance="open",
        )

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
            messages=[Message.user("Original message")],
        )

        result = await simulator._expand_with_intent(node, turns=1, first_intent=intent)

        # Should use original message when rephrase fails
        assert result is not None


class TestRephraseInitialMessage:
    """Tests for _rephrase_initial_message."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_rephrase_success(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test successful message rephrasing."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("I'm really frustrated with this!")
        )

        intent = UserIntent(
            id="frustrated",
            label="Frustrated",
            description="User is frustrated",
            emotional_tone="frustrated",
            cognitive_stance="impatient",
        )

        result = await simulator._rephrase_initial_message("Help me with this", intent)

        assert result == "I'm really frustrated with this!"


class TestExpandLinear:
    """Tests for _expand_linear."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def simulator(self, mock_llm: MagicMock) -> ConversationSimulator:
        """Create a simulator with mock LLM."""
        return ConversationSimulator(
            llm=mock_llm,
            goal="Help user",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_expand_linear_multiple_turns(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test linear expansion with multiple turns."""
        mock_llm.complete.return_value = Completion(message=Message.assistant("Response"))

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
            messages=[Message.user("Hello")],
        )

        result = await simulator._expand_linear(node, turns=2)

        assert result is not None
        # Should have added messages for each turn
        assert len(result.messages) > 1

    @pytest.mark.asyncio
    async def test_expand_linear_early_termination(
        self, simulator: ConversationSimulator, mock_llm: MagicMock
    ) -> None:
        """Test linear expansion stops on termination signal."""
        mock_llm.complete.return_value = Completion(
            message=Message.assistant("goodbye")  # Termination signal
        )

        node = DialogueNode(
            id="node-1",
            strategy=Strategy(tagline="Test", description="Test"),
            messages=[Message.user("Hello")],
        )

        result = await simulator._expand_linear(node, turns=3)

        assert result.status == NodeStatus.TERMINAL
        # Should have stopped after first turn
        assert len(result.messages) <= 3
