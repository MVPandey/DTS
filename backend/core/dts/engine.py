"""Dialogue Tree Search Engine - Main orchestrator."""

from __future__ import annotations

import logging

from backend.core.dts.config import DTSConfig
from backend.core.dts.components.evaluator import TrajectoryEvaluator
from backend.core.dts.components.generator import StrategyGenerator
from backend.core.dts.components.simulator import ConversationSimulator
from backend.core.dts.tree import DialogueTree, generate_node_id
from backend.core.dts.types import (
    AggregatedScore,
    DialogueNode,
    DTSRunResult,
    NodeStatus,
    TokenTracker,
)
from backend.llm.client import LLM
from backend.llm.types import Completion, Message

logger = logging.getLogger(__name__)


def _log(phase: str, message: str, indent: int = 0) -> None:
    """Print a formatted log message."""
    prefix = "  " * indent
    print(f"[DTS:{phase}] {prefix}{message}")


class DTSEngine:
    """
    Dialogue Tree Search Engine.

    Explores multiple conversation branches in parallel, scores them with
    judges, and prunes to focus on promising paths.

    This is a parallel beam search algorithm optimized for dialogue:
    - Generates diverse conversation strategies
    - Forks branches based on user intent diversity
    - Evaluates trajectories with multiple judges
    - Prunes low-scoring paths to focus computation

    Usage:
        engine = DTSEngine(
            llm=llm,
            config=DTSConfig(
                goal="Help user debug code",
                first_message="I'm having trouble with Python",
            ),
        )
        result = await engine.run(rounds=2)
    """

    def __init__(
        self,
        llm: LLM,
        config: DTSConfig,
    ) -> None:
        """
        Initialize the DTS engine.

        Args:
            llm: LLM client for all operations.
            config: Configuration for the search.
        """
        self.llm = llm
        self.config = config

        # Auto-detect model from LLM client if not provided
        model = config.model or getattr(llm, "_default_model", None)

        # Token tracking
        self._token_tracker = TokenTracker(model_name=model or "unknown")

        # Create components with shared token tracking
        self._generator = StrategyGenerator(
            llm=llm,
            goal=config.goal,
            model=model,
            temperature=config.temperature,
            max_concurrency=config.max_concurrency,
            on_usage=self._track_usage,
        )

        self._simulator = ConversationSimulator(
            llm=llm,
            goal=config.goal,
            model=model,
            temperature=config.temperature,
            max_concurrency=config.max_concurrency,
            on_usage=self._track_usage,
        )

        self._evaluator = TrajectoryEvaluator(
            llm=llm,
            goal=config.goal,
            model=model,
            judge_temperature=config.judge_temperature,
            prune_threshold=config.prune_threshold,
            max_concurrency=config.max_concurrency,
            on_usage=self._track_usage,
        )

        self._tree: DialogueTree | None = None

    async def run(self, rounds: int = 1) -> DTSRunResult:
        """
        Execute the dialogue tree search.

        Args:
            rounds: Number of expansion/pruning rounds.

        Returns:
            DTSRunResult with best trajectory and statistics.
        """
        cfg = self.config

        print("\n" + "=" * 60)
        print("DIALOGUE TREE SEARCH - Starting")
        print("=" * 60)
        _log("INIT", f"Goal: {cfg.goal[:50]}...")
        _log(
            "INIT",
            f"Branches: {cfg.init_branches} | Turns: {cfg.turns_per_branch} | Rounds: {rounds}",
        )
        if cfg.user_intents_per_branch > 1:
            _log(
                "INIT",
                f"User intent forking: {cfg.user_intents_per_branch} intents/branch",
            )
        _log("INIT", f"Scoring mode: {cfg.scoring_mode}")

        # Initialize tree
        _log("INIT", "Creating tree structure...")
        tree = await self._initialize_tree()
        self._tree = tree

        total_pruned = 0

        for round_num in range(rounds):
            print("\n" + "-" * 40)
            _log("ROUND", f"Round {round_num + 1}/{rounds}")
            print("-" * 40)

            # Get expandable leaves
            active_leaves = tree.active_leaves()
            if not active_leaves:
                logger.warning("No active leaves")
                break

            expandable = [n for n in active_leaves if n.strategy is not None]
            if not expandable:
                logger.warning("No expandable nodes")
                break

            # Expand branches
            _log(
                "EXPAND",
                f"Expanding {len(expandable)} branches ({cfg.turns_per_branch} turns)...",
            )
            expanded = await self._simulator.expand_nodes(
                expandable,
                turns=cfg.turns_per_branch,
                intents_per_node=cfg.user_intents_per_branch,
                tree=tree,
                generate_intents=self._generator.generate_intents,
            )
            _log("EXPAND", f"Completed {len(expanded)} expansions", indent=1)

            # Score branches
            if cfg.scoring_mode == "comparative":
                _log("JUDGE", f"Comparative ranking of {len(expanded)} branches...")
                scores = await self._evaluator.evaluate_comparative(expanded)
            else:
                _log("JUDGE", f"Scoring {len(expanded)} branches (3 judges each)...")
                scores = await self._evaluator.evaluate_absolute(expanded)

            # Log scores
            for node in expanded:
                if node.id in scores:
                    score = scores[node.id]
                    strategy = node.strategy.tagline if node.strategy else "unknown"
                    intent = f" [{node.user_intent.label}]" if node.user_intent else ""
                    _log(
                        "JUDGE",
                        f"'{strategy}'{intent}: {score.aggregated_score:.1f}/10",
                        indent=1,
                    )

            # Backpropagate
            for node in expanded:
                if node.id in scores:
                    tree.backpropagate(node.id, scores[node.id].aggregated_score)

            # Prune
            _log("PRUNE", f"Pruning (threshold: {cfg.prune_threshold})...")
            survivors = self._prune(expanded, scores)
            pruned_count = len(expanded) - len(survivors)
            total_pruned += pruned_count
            _log("PRUNE", f"Kept {len(survivors)}, pruned {pruned_count}", indent=1)

            for node in survivors:
                strategy = node.strategy.tagline if node.strategy else "unknown"
                intent = f" [{node.user_intent.label}]" if node.user_intent else ""
                _log("PRUNE", f"Survivor: '{strategy}'{intent}", indent=2)

        # Find best
        best_node = tree.best_leaf_by_score()

        print("\n" + "=" * 60)
        _log("DONE", "Search complete!")
        if best_node:
            best_strategy = best_node.strategy.tagline if best_node.strategy else "root"
            _log(
                "DONE",
                f"Best: '{best_strategy}' with score {best_node.stats.aggregated_score:.1f}/10",
            )
        print("=" * 60 + "\n")

        self._token_tracker.print_summary()

        return DTSRunResult(
            best_node_id=best_node.id if best_node else None,
            best_score=best_node.stats.aggregated_score if best_node else 0.0,
            best_messages=list(best_node.messages) if best_node else [],
            all_nodes=tree.all_nodes(),
            pruned_count=total_pruned,
            token_usage=self._token_tracker.to_dict(),
            total_rounds=rounds,
        )

    async def _initialize_tree(self) -> DialogueTree:
        """Initialize tree with root and initial strategy branches."""
        cfg = self.config

        # Create root
        root = DialogueNode(
            id=generate_node_id(),
            depth=0,
            messages=[Message.user(cfg.first_message)],
        )
        tree = DialogueTree.create(root)

        # Generate strategies
        _log("INIT", "Generating strategies...", indent=1)
        deep_context = self._get_deep_research_context()
        strategies = await self._generator.generate_strategies(
            cfg.first_message,
            cfg.init_branches,
            deep_context,
        )
        _log("INIT", f"Generated {len(strategies)} strategies:", indent=1)

        for i, strategy in enumerate(strategies, 1):
            _log("INIT", f"{i}. {strategy.tagline}", indent=2)

        # Create children
        for strategy in strategies:
            child = DialogueNode(
                id=generate_node_id(),
                strategy=strategy,
                messages=[Message.user(cfg.first_message)],
            )
            tree.add_child(root.id, child)

        _log("INIT", f"Tree initialized with {len(strategies)} branches", indent=1)
        return tree

    def _prune(
        self,
        nodes: list[DialogueNode],
        scores: dict[str, AggregatedScore],
    ) -> list[DialogueNode]:
        """Prune low-scoring branches."""
        cfg = self.config

        if not nodes:
            return []

        # Threshold filter
        survivors = [
            n
            for n in nodes
            if n.id in scores and scores[n.id].aggregated_score >= cfg.prune_threshold
        ]

        # Top-K cap
        if cfg.keep_top_k and len(survivors) > cfg.keep_top_k:
            survivors.sort(
                key=lambda n: scores[n.id].aggregated_score,
                reverse=True,
            )
            survivors = survivors[: cfg.keep_top_k]

        # Min survivors
        if len(survivors) < cfg.min_survivors:
            ranked = sorted(
                nodes,
                key=lambda n: scores.get(n.id, self._zero_score()).aggregated_score,
                reverse=True,
            )
            survivors = ranked[: cfg.min_survivors]

        # Mark pruned
        survivor_ids = {n.id for n in survivors}
        for n in nodes:
            if n.id not in survivor_ids:
                n.status = NodeStatus.PRUNED
                score = scores.get(n.id)
                if score:
                    n.prune_reason = (
                        f"score {score.aggregated_score:.1f} < {cfg.prune_threshold}"
                    )
                else:
                    n.prune_reason = "scoring failed"

        return survivors

    def _zero_score(self) -> AggregatedScore:
        """Create a zero score for fallback."""
        return AggregatedScore(
            individual_scores=[0, 0, 0],
            aggregated_score=0,
            pass_threshold=self.config.prune_threshold,
            pass_votes=0,
            passed=False,
        )

    def _get_deep_research_context(self) -> str | None:
        """Get deep research context (stub)."""
        if not self.config.deep_research:
            return None
        return "Relevant domain research context available."

    def _track_usage(self, completion: Completion, phase: str) -> None:
        """Track token usage by phase."""
        if not completion.usage:
            return

        tracker = self._token_tracker
        if phase == "strategy":
            tracker.strategy_generation.add(completion.usage)
        elif phase == "intent":
            tracker.intent_generation.add(completion.usage)
        elif phase == "user":
            tracker.user_simulation.add(completion.usage)
        elif phase == "assistant":
            tracker.assistant_generation.add(completion.usage)
        elif phase == "judge":
            tracker.judging.add(completion.usage)

    @property
    def tree(self) -> DialogueTree | None:
        """Get the current tree (available after run())."""
        return self._tree
