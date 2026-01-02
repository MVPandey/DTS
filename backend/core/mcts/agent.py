"""MCTS Agent for conversational AI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.core.mcts.aggregator import aggregate_majority_vote
from backend.core.mcts.tree import MCTSTree, generate_node_id
from backend.core.mcts.types import (
    AggregatedScore,
    BranchStrategy,
    MCTSNode,
    MCTSRunResult,
    NodeStatus,
)
from backend.core.prompts import prompts
from backend.llm.client import LLM
from backend.llm.types import Message

logger = logging.getLogger(__name__)


def _log(phase: str, message: str, indent: int = 0) -> None:
    """Print a formatted log message with phase indicator."""
    prefix = "  " * indent
    print(f"[MCTS:{phase}] {prefix}{message}")


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for conversational AI.

    Explores multiple conversation branches in parallel, scores them with
    multiple judges, and prunes to focus on promising paths.

    Usage:
        agent = MCTSAgent(
            llm=llm,
            goal="Help user debug their code",
            first_message="I'm having trouble with my Python script",
            init_branch=6,
            deep_research=False,
        )
        result = await agent.run(rounds=2)
        print(result.best_messages)
    """

    def __init__(
        self,
        *,
        llm: LLM,
        goal: str,
        first_message: str,
        init_branch: int = 6,
        deep_research: bool = False,
        turns_per_branch: int = 5,
        prune_threshold: float = 5.0,
        keep_top_k: int | None = None,
        min_survivors: int = 1,
        max_concurrency: int = 16,
        model: str | None = None,
        temperature: float = 0.7,
        judge_temperature: float = 0.3,
    ) -> None:
        """
        Initialize the MCTS agent.

        Args:
            llm: LLM client for completions.
            goal: Conversation goal/objective.
            first_message: Initial user message to start the conversation.
            init_branch: Number of initial branches to create.
            deep_research: Whether to include deep research context.
            turns_per_branch: Number of turns (user+assistant) per expansion.
            prune_threshold: Score threshold for pruning (0-10).
            keep_top_k: Keep only top K branches after pruning (optional).
            min_survivors: Minimum branches to keep even if below threshold.
            max_concurrency: Maximum concurrent LLM calls.
            model: Model to use for completions.
            temperature: Temperature for conversation generation.
            judge_temperature: Temperature for judge evaluations (lower = more deterministic).
        """
        self.llm = llm
        self.goal = goal
        self.first_message = first_message
        self.init_branch = init_branch
        self.deep_research = deep_research
        self.turns_per_branch = turns_per_branch
        self.prune_threshold = prune_threshold
        self.keep_top_k = keep_top_k
        self.min_survivors = min_survivors
        self.max_concurrency = max_concurrency
        self.model = model
        self.temperature = temperature
        self.judge_temperature = judge_temperature

        self._sem = asyncio.Semaphore(max_concurrency)
        self._tree: MCTSTree | None = None

    async def run(self, rounds: int = 1) -> MCTSRunResult:
        """
        Execute the MCTS search.

        Args:
            rounds: Number of expansion/pruning rounds.

        Returns:
            MCTSRunResult with best trajectory and tree statistics.
        """
        print("\n" + "=" * 60)
        print("MCTS AGENT - Starting Search")
        print("=" * 60)
        _log("INIT", f"Goal: {self.goal[:50]}...")
        _log(
            "INIT",
            f"Branches: {self.init_branch} | Turns: {self.turns_per_branch} | Rounds: {rounds}",
        )

        # Initialize tree with root and initial branches
        _log("INIT", "Creating tree structure...")
        tree = await self._initialize_tree()
        self._tree = tree

        total_pruned = 0

        for round_num in range(rounds):
            print("\n" + "-" * 40)
            _log("ROUND", f"Round {round_num + 1}/{rounds}")
            print("-" * 40)

            # Get active leaves to expand
            active_leaves = tree.active_leaves()
            if not active_leaves:
                logger.warning("No active leaves to expand")
                break

            # Skip root node (it has no strategy)
            expandable = [n for n in active_leaves if n.strategy is not None]
            if not expandable:
                logger.warning("No expandable nodes (no strategies)")
                break

            # Expand all branches in parallel
            _log(
                "EXPAND",
                f"Expanding {len(expandable)} branches ({self.turns_per_branch} turns each)...",
            )
            expanded_nodes = await self._expand_branches_parallel(
                expandable, turns=self.turns_per_branch
            )
            _log("EXPAND", f"Completed {len(expanded_nodes)} expansions", indent=1)

            # Score all expanded branches with 3 judges each
            _log("JUDGE", f"Scoring {len(expanded_nodes)} branches (3 judges each)...")
            scores_by_id = await self._judge_branches_parallel(expanded_nodes)

            # Log scores
            for node in expanded_nodes:
                if node.id in scores_by_id:
                    score = scores_by_id[node.id]
                    strategy_name = (
                        node.strategy.tagline if node.strategy else "unknown"
                    )
                    _log(
                        "JUDGE",
                        f"'{strategy_name}': {score.aggregated_score:.1f}/10 (votes: {score.individual_scores})",
                        indent=1,
                    )

            # Backpropagate scores
            for node in expanded_nodes:
                if node.id in scores_by_id:
                    score = scores_by_id[node.id].aggregated_score
                    tree.backpropagate(node.id, score)

            # Prune low-scoring branches
            _log("PRUNE", f"Pruning (threshold: {self.prune_threshold})...")
            survivors = self._prune(expanded_nodes, scores_by_id)
            pruned_count = len(expanded_nodes) - len(survivors)
            total_pruned += pruned_count
            _log(
                "PRUNE",
                f"Kept {len(survivors)} branches, pruned {pruned_count}",
                indent=1,
            )

            for node in survivors:
                strategy_name = node.strategy.tagline if node.strategy else "unknown"
                _log("PRUNE", f"Survivor: '{strategy_name}'", indent=2)

        # Find best result
        best_node = tree.best_leaf_by_score()

        print("\n" + "=" * 60)
        _log("DONE", "Search complete!")
        if best_node:
            best_strategy = best_node.strategy.tagline if best_node.strategy else "root"
            _log(
                "DONE",
                f"Best branch: '{best_strategy}' with score {best_node.stats.aggregated_score:.1f}/10",
            )
        print("=" * 60 + "\n")

        return MCTSRunResult(
            best_node_id=best_node.id if best_node else None,
            best_score=best_node.stats.aggregated_score if best_node else 0.0,
            best_messages=list(best_node.messages) if best_node else [],
            all_nodes=tree.all_nodes(),
            pruned_count=total_pruned,
            total_rounds=rounds,
        )

    async def _initialize_tree(self) -> MCTSTree:
        """Initialize tree with root node and generate initial branches."""
        # Create root node
        root = MCTSNode(
            id=generate_node_id(),
            depth=0,
            messages=[Message.user(self.first_message)],
        )
        tree = MCTSTree.create(root)

        # Generate initial branch strategies
        _log("INIT", "Generating branch strategies...", indent=1)
        strategies = await self._generate_initial_branches()
        _log("INIT", f"Generated {len(strategies)} strategies:", indent=1)

        for i, strategy in enumerate(strategies, 1):
            _log("INIT", f"{i}. {strategy.tagline}", indent=2)

        # Create child nodes for each strategy
        for strategy in strategies:
            child = MCTSNode(
                id=generate_node_id(),
                strategy=strategy,
                messages=[Message.user(self.first_message)],
            )
            tree.add_child(root.id, child)

        _log("INIT", f"Tree initialized with {len(strategies)} branches", indent=1)
        return tree

    async def _generate_initial_branches(self) -> list[BranchStrategy]:
        """Generate initial branch strategies using the tree generator prompt."""
        deep_research_context = self._get_deep_research_context()

        prompt = prompts.conversation_tree_generator(
            num_nodes=self.init_branch,
            conversation_goal=self.goal,
            conversation_context=self.first_message,
            deep_research_context=deep_research_context,
        )

        completion = await self._call_llm_json(prompt)

        if not completion:
            logger.error("Failed to generate initial branches")
            return []

        strategies = []
        nodes_data = completion.get("nodes", {})

        for tagline, description in nodes_data.items():
            strategies.append(
                BranchStrategy(tagline=tagline, description=str(description))
            )

        return strategies

    async def _expand_branches_parallel(
        self,
        nodes: list[MCTSNode],
        turns: int,
    ) -> list[MCTSNode]:
        """Expand all branches in parallel."""
        tasks = [self._expand_branch(node, turns) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        expanded = []
        for node, result in zip(nodes, results):
            if isinstance(result, Exception):
                logger.error(f"Error expanding node {node.id}: {result}")
                node.status = NodeStatus.ERROR
            else:
                expanded.append(result)

        return expanded

    async def _expand_branch(self, node: MCTSNode, turns: int) -> MCTSNode:
        """
        Expand a branch by simulating User + Assistant conversation for n turns.

        Each turn consists of:
        1. Simulate user response
        2. Generate assistant response (goal-directed, following strategy)
        """
        history = list(node.messages)

        for _ in range(turns):
            # A. Simulate user response
            user_response = await self._simulate_user(
                history, conversation_goal=self.goal
            )
            history.append(Message.user(user_response))
            _log("EXPAND", f"User response: {user_response}", indent=2)

            # B. Generate assistant response following strategy
            assistant_response = await self._generate_assistant(history, node.strategy)
            _log("EXPAND", f"Assistant response: {assistant_response}", indent=2)
            history.append(Message.assistant(assistant_response))

        # Update node with expanded trajectory
        node.messages = history
        return node

    async def _simulate_user(
        self, history: list[Message], conversation_goal: str
    ) -> str:
        system_prompt = prompts.user_simulation(
            conversation_goal=conversation_goal,
            conversation_history=self._format_history(history),
        )
        messages = [Message.system(system_prompt)] + history

        completion = await self._call_llm(messages)
        return completion.message.content or ""

    async def _generate_assistant(
        self,
        history: list[Message],
        strategy: BranchStrategy | None,
    ) -> str:
        """Generate an assistant response following the strategy."""
        system_prompt = prompts.assistant_continuation(
            conversation_goal=self.goal,
            conversation_history=self._format_history(history),
            strategy_tagline=strategy.tagline if strategy else "",
            strategy_description=strategy.description if strategy else "",
        )

        messages = [Message.system(system_prompt)] + history
        completion = await self._call_llm(messages)
        return completion.message.content or ""

    async def _judge_branches_parallel(
        self,
        nodes: list[MCTSNode],
    ) -> dict[str, AggregatedScore]:
        """Score all branches with 3 parallel judges each."""
        tasks = [self._judge_branch(node) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores_by_id: dict[str, AggregatedScore] = {}

        for node, result in zip(nodes, results):
            if isinstance(result, Exception):
                logger.error(f"Error judging node {node.id}: {result}")
                # Assign minimum score on error
                scores_by_id[node.id] = AggregatedScore(
                    individual_scores=[0.0, 0.0, 0.0],
                    aggregated_score=0.0,
                    pass_threshold=self.prune_threshold,
                    pass_votes=0,
                    passed=False,
                )
            else:
                scores_by_id[node.id] = result
                # Update node stats
                node.stats.judge_scores = result.individual_scores
                node.stats.aggregated_score = result.aggregated_score

        return scores_by_id

    async def _judge_branch(self, node: MCTSNode) -> AggregatedScore:
        """Run 3 parallel judges on a branch and aggregate scores."""
        # Format conversation history for judge
        history_str = self._format_history(node.messages)

        prompt = prompts.trajectory_outcome_judge(
            conversation_goal=self.goal,
            conversation_history=history_str,
        )

        # Run 3 judges in parallel
        tasks = [
            self._call_llm_json(prompt, temperature=self.judge_temperature)
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        _log("JUDGE", f"Results: {results}", indent=2)

        scores: list[float] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Judge failed: {result}")
                scores.append(0.0)
            elif result and "total_score" in result:
                scores.append(float(result["total_score"]))
            else:
                scores.append(0.0)

        # Ensure we have exactly 3 scores
        while len(scores) < 3:
            scores.append(0.0)

        return aggregate_majority_vote(scores[:3], pass_threshold=self.prune_threshold)

    def _prune(
        self,
        nodes: list[MCTSNode],
        scores_by_id: dict[str, AggregatedScore],
    ) -> list[MCTSNode]:
        """
        Prune low-scoring branches.

        Strategy:
        1. Keep nodes with aggregated_score >= prune_threshold
        2. If keep_top_k set, limit to top K
        3. Always keep at least min_survivors
        """
        if not nodes:
            return []

        # Step 1: Threshold filter
        survivors = [
            n
            for n in nodes
            if n.id in scores_by_id
            and scores_by_id[n.id].aggregated_score >= self.prune_threshold
        ]

        # Step 2: Top-K cap (optional)
        if self.keep_top_k and len(survivors) > self.keep_top_k:
            survivors.sort(
                key=lambda n: scores_by_id[n.id].aggregated_score,
                reverse=True,
            )
            survivors = survivors[: self.keep_top_k]

        # Step 3: Min survivors safety
        if len(survivors) < self.min_survivors:
            ranked = sorted(
                nodes,
                key=lambda n: scores_by_id.get(
                    n.id,
                    AggregatedScore(
                        individual_scores=[0, 0, 0],
                        aggregated_score=0,
                        pass_threshold=self.prune_threshold,
                        pass_votes=0,
                        passed=False,
                    ),
                ).aggregated_score,
                reverse=True,
            )
            survivors = ranked[: self.min_survivors]

        # Mark pruned nodes
        survivor_ids = {n.id for n in survivors}
        for n in nodes:
            if n.id not in survivor_ids:
                n.status = NodeStatus.PRUNED
                score = scores_by_id.get(n.id)
                if score:
                    n.prune_reason = (
                        f"score {score.aggregated_score:.1f} < {self.prune_threshold}"
                    )
                else:
                    n.prune_reason = "scoring failed"

        return survivors

    def _get_deep_research_context(self) -> str | None:
        """Get deep research context (stub for now)."""
        if not self.deep_research:
            return None
        # TODO: Implement retrieval pipeline
        return "Relevant domain research context available."

    def _format_history(self, messages: list[Message]) -> str:
        """Format message history as a string for prompts."""
        lines = []
        for msg in messages:
            role = msg.role.capitalize()
            content = msg.content or ""
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    async def _call_llm(
        self,
        messages: list[Message],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Make an LLM call with concurrency control."""
        async with self._sem:
            return await self.llm.complete(
                messages,
                model=self.model,
                temperature=temperature or self.temperature,
                **kwargs,
            )

    async def _call_llm_json(
        self,
        prompt: str,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Make an LLM call expecting JSON output."""
        async with self._sem:
            try:
                completion = await self.llm.complete(
                    [Message.user(prompt)],
                    model=self.model,
                    temperature=temperature or self.temperature,
                    structured_output=True,
                    **kwargs,
                )
                return completion.data
            except Exception as e:
                logger.error(f"JSON LLM call failed: {e}")
                return None

    @property
    def tree(self) -> MCTSTree | None:
        """Get the current tree (available after run())."""
        return self._tree
