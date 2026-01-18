"""Dialogue Tree Search Engine - Main orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.core.dts.components.evaluator import TrajectoryEvaluator
from backend.core.dts.components.generator import FIXED_INTENT, StrategyGenerator
from backend.core.dts.components.researcher import DeepResearcher
from backend.core.dts.components.simulator import ConversationSimulator
from backend.core.dts.config import DTSConfig
from backend.core.dts.tree import DialogueTree, generate_node_id
from backend.core.dts.types import (
    AggregatedScore,
    DialogueNode,
    DTSRunResult,
    NodeStatus,
    TokenTracker,
)
from backend.core.dts.utils import emit_event, log_phase
from backend.llm.client import LLM
from backend.llm.types import Completion, Message
from backend.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable

    EventCallback = Callable[[str, dict], Awaitable[None]]


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

        # Resolve per-phase models with fallbacks
        default_model = config.model or getattr(llm, "_default_model", None)
        strategy_model = config.strategy_model or default_model
        simulator_model = config.simulator_model or default_model
        judge_model = config.judge_model or default_model

        # Token tracking (use default model name for summary)
        self._token_tracker = TokenTracker(model_name=default_model or "unknown")

        # Create components with per-phase models
        self._generator = StrategyGenerator(
            llm=llm,
            goal=config.goal,
            model=strategy_model,
            temperature=config.temperature,
            max_concurrency=config.max_concurrency,
            on_usage=self._track_usage,
            provider=config.provider,
            reasoning_enabled=config.reasoning_enabled,
        )

        self._simulator = ConversationSimulator(
            llm=llm,
            goal=config.goal,
            model=simulator_model,
            temperature=config.temperature,
            max_concurrency=config.max_concurrency,
            on_usage=self._track_usage,
            on_event=self._emit,
            provider=config.provider,
            reasoning_enabled=config.reasoning_enabled,
        )

        self._evaluator = TrajectoryEvaluator(
            llm=llm,
            goal=config.goal,
            model=judge_model,
            judge_temperature=config.judge_temperature,
            prune_threshold=config.prune_threshold,
            max_concurrency=config.max_concurrency,
            on_usage=self._track_usage,
            provider=config.provider,
            reasoning_enabled=config.reasoning_enabled,
        )

        self._researcher = DeepResearcher(
            llm=llm,
            model=strategy_model,  # Use strategy model for query distillation
            cache_dir=config.research_cache_dir,
            on_cost=self._track_research_cost,
            on_event=self._emit,
        )

        self._tree: DialogueTree | None = None
        self._event_callback: EventCallback | None = None
        self._research_report: str | None = None

    def set_event_callback(self, callback: EventCallback) -> None:
        """
        Set a callback for receiving real-time events during the run.

        The callback receives (event_type, data) and should be async.
        Event types:
        - "round_started": { round, total_rounds }
        - "node_added": { node data }
        - "node_updated": { id, status, score? }
        - "nodes_pruned": { ids, reasons }
        - "token_update": { totals }
        """
        self._event_callback = callback

    async def run(self, rounds: int = 1) -> DTSRunResult:
        """
        Execute the dialogue tree search.

        Args:
            rounds: Number of expansion/pruning rounds.

        Returns:
            DTSRunResult with best trajectory and statistics.
        """
        cfg = self.config

        logger.info("=" * 60)
        logger.info("DIALOGUE TREE SEARCH - Starting")
        logger.info("=" * 60)
        log_phase(logger, "INIT", f"Goal: {cfg.goal[:50]}...")
        log_phase(
            logger,
            "INIT",
            f"Branches: {cfg.init_branches} | Turns: {cfg.turns_per_branch} | Rounds: {rounds}",
        )
        if cfg.user_intents_per_branch > 1:
            log_phase(
                logger,
                "INIT",
                f"User intent forking: {cfg.user_intents_per_branch} intents/branch",
            )
        log_phase(logger, "INIT", f"Scoring mode: {cfg.scoring_mode}")

        # Emit search started event
        self._emit(
            "search_started",
            {
                "goal": cfg.goal,
                "first_message": cfg.first_message,
                "total_rounds": rounds,
                "config": {
                    "init_branches": cfg.init_branches,
                    "turns_per_branch": cfg.turns_per_branch,
                    "user_intents_per_branch": cfg.user_intents_per_branch,
                    "scoring_mode": cfg.scoring_mode,
                    "prune_threshold": cfg.prune_threshold,
                },
            },
        )

        # Initialize tree
        log_phase(logger, "INIT", "Creating tree structure...")
        self._emit("phase", {"phase": "initializing", "message": "Creating tree structure..."})
        tree = await self._initialize_tree()
        self._tree = tree

        total_pruned = 0

        for round_num in range(rounds):
            logger.info("-" * 40)
            log_phase(logger, "ROUND", f"Round {round_num + 1}/{rounds}")
            logger.info("-" * 40)

            # Emit round started event
            self._emit(
                "round_started",
                {"round": round_num + 1, "total_rounds": rounds},
            )

            # Get expandable leaves
            active_leaves = tree.active_leaves()
            if not active_leaves:
                logger.warning("No active leaves")
                break

            expandable = [n for n in active_leaves if n.strategy is not None]
            if not expandable:
                logger.warning("No expandable nodes")
                break

            # Emit intent generation phase if forking
            if cfg.user_intents_per_branch > 1:
                log_phase(
                    logger,
                    "INTENT",
                    f"Generating {cfg.user_intents_per_branch} user intents per branch...",
                )
                self._emit(
                    "phase",
                    {
                        "phase": "generating_intents",
                        "message": f"Generating {cfg.user_intents_per_branch} user intents per branch...",
                        "intents_per_branch": cfg.user_intents_per_branch,
                        "branch_count": len(expandable),
                    },
                )

            # Expand branches
            log_phase(
                logger,
                "EXPAND",
                f"Expanding {len(expandable)} branches ({cfg.turns_per_branch} turns)...",
            )
            self._emit(
                "phase",
                {
                    "phase": "expanding",
                    "message": f"Expanding {len(expandable)} branches...",
                    "branch_count": len(expandable),
                    "turns_per_branch": cfg.turns_per_branch,
                },
            )
            # Determine intent generation strategy
            if cfg.user_variability:
                # Generate diverse user intents via LLM
                intents_per_node = cfg.user_intents_per_branch
                generate_intents_fn = self._generator.generate_intents
            else:
                # Use fixed "healthily critical + engaged" persona (no API call)
                intents_per_node = 1

                async def fixed_intent_fn(_history: list, _count: int) -> list:
                    return [FIXED_INTENT]

                generate_intents_fn = fixed_intent_fn

            expanded = await self._simulator.expand_nodes(
                expandable,
                turns=cfg.turns_per_branch,
                intents_per_node=intents_per_node,
                tree=tree,
                generate_intents=generate_intents_fn,
            )
            log_phase(logger, "EXPAND", f"Completed {len(expanded)} expansions", indent=1)

            # Emit node_added events for expanded nodes
            for node in expanded:
                self._emit(
                    "node_added",
                    {
                        "id": node.id,
                        "parent_id": node.parent_id,
                        "depth": node.depth,
                        "status": node.status.value,
                        "strategy": node.strategy.tagline if node.strategy else None,
                        "user_intent": node.intent_label,
                        "message_count": len(node.messages),
                    },
                )

            # Score branches
            self._emit(
                "phase",
                {
                    "phase": "scoring",
                    "message": f"Scoring {len(expanded)} branches...",
                    "node_count": len(expanded),
                    "scoring_mode": cfg.scoring_mode,
                },
            )
            if cfg.scoring_mode == "comparative":
                log_phase(
                    logger,
                    "JUDGE",
                    f"Comparative ranking of {len(expanded)} branches...",
                )
                scores = await self._evaluator.evaluate_comparative(expanded)
            else:
                log_phase(
                    logger,
                    "JUDGE",
                    f"Scoring {len(expanded)} branches (3 judges each)...",
                )
                scores = await self._evaluator.evaluate_absolute(expanded)

            # Log scores and emit events
            for node in expanded:
                if node.id in scores:
                    score = scores[node.id]
                    intent_str = f" [{node.intent_label}]" if node.intent_label else ""
                    log_phase(
                        logger,
                        "JUDGE",
                        f"'{node.strategy_label}'{intent_str}: {score.aggregated_score:.1f}/10",
                        indent=1,
                    )
                    # Emit score update
                    self._emit(
                        "node_updated",
                        {
                            "id": node.id,
                            "status": "scored",
                            "score": score.aggregated_score,
                            "individual_scores": score.individual_scores,
                            "passed": score.passed,
                        },
                    )

            # Backpropagate
            for node in expanded:
                if node.id in scores:
                    tree.backpropagate(node.id, scores[node.id].aggregated_score)

            # Prune
            log_phase(logger, "PRUNE", f"Pruning (threshold: {cfg.prune_threshold})...")
            self._emit(
                "phase",
                {
                    "phase": "pruning",
                    "message": f"Pruning branches below {cfg.prune_threshold}...",
                    "threshold": cfg.prune_threshold,
                },
            )
            survivors = self._prune(expanded, scores)
            pruned_count = len(expanded) - len(survivors)
            total_pruned += pruned_count
            log_phase(
                logger,
                "PRUNE",
                f"Kept {len(survivors)}, pruned {pruned_count}",
                indent=1,
            )

            # Emit pruning event
            pruned_nodes = [n for n in expanded if n.status == NodeStatus.PRUNED]
            if pruned_nodes:
                self._emit(
                    "nodes_pruned",
                    {
                        "ids": [n.id for n in pruned_nodes],
                        "reasons": {n.id: n.prune_reason for n in pruned_nodes},
                    },
                )

            # Emit token update
            self._emit(
                "token_update",
                {
                    "totals": {
                        "input_tokens": self._token_tracker.total_input_tokens,
                        "output_tokens": self._token_tracker.total_output_tokens,
                        "total_cost_usd": round(self._token_tracker.total_cost, 6),
                    }
                },
            )

            for node in survivors:
                intent_str = f" [{node.intent_label}]" if node.intent_label else ""
                log_phase(
                    logger,
                    "PRUNE",
                    f"Survivor: '{node.strategy_label}'{intent_str}",
                    indent=2,
                )

        # Find best
        best_node = tree.best_leaf_by_score()

        logger.info("=" * 60)
        log_phase(logger, "DONE", "Search complete!")
        if best_node:
            log_phase(
                logger,
                "DONE",
                f"Best: '{best_node.strategy_label}' with score {best_node.stats.aggregated_score:.1f}/10",
            )
        logger.info("=" * 60)

        self._token_tracker.print_summary()

        # Emit completion event
        self._emit(
            "phase",
            {
                "phase": "complete",
                "message": "Search complete!",
                "best_score": best_node.stats.aggregated_score if best_node else 0.0,
                "best_strategy": best_node.strategy_label if best_node else None,
            },
        )

        return DTSRunResult(
            best_node_id=best_node.id if best_node else None,
            best_score=best_node.stats.aggregated_score if best_node else 0.0,
            best_messages=list(best_node.messages) if best_node else [],
            all_nodes=tree.all_nodes(),
            pruned_count=total_pruned,
            token_usage=self._token_tracker.to_dict(),
            total_rounds=rounds,
            research_report=self._research_report,
        )

    def _emit(self, event_type: str, data: dict) -> None:
        """Emit an event if callback is set (fire-and-forget)."""
        if self._event_callback is not None:
            asyncio.create_task(emit_event(self._event_callback, event_type, data, logger))

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

        # Emit root node
        self._emit(
            "node_added",
            {
                "id": root.id,
                "parent_id": None,
                "depth": 0,
                "status": root.status.value,
                "strategy": None,
                "user_intent": None,
                "message_count": len(root.messages),
            },
        )

        # Deep research if enabled
        if cfg.deep_research:
            log_phase(logger, "INIT", "Conducting deep research...", indent=1)
            self._emit(
                "phase",
                {
                    "phase": "researching",
                    "message": "Conducting deep research on the topic...",
                },
            )
        deep_context = await self._get_deep_research_context()

        # Pass research context to evaluator for informed judging
        if deep_context:
            self._evaluator.set_research_context(deep_context)

        # Generate strategies
        log_phase(logger, "INIT", "Generating strategies...", indent=1)
        self._emit(
            "phase",
            {
                "phase": "generating_strategies",
                "message": f"Generating {cfg.init_branches} conversation strategies...",
                "count": cfg.init_branches,
            },
        )
        strategies = await self._generator.generate_strategies(
            cfg.first_message,
            cfg.init_branches,
            deep_context,
        )
        log_phase(logger, "INIT", f"Generated {len(strategies)} strategies:", indent=1)

        for i, strategy in enumerate(strategies, 1):
            log_phase(logger, "INIT", f"{i}. {strategy.tagline}", indent=2)
            self._emit(
                "strategy_generated",
                {
                    "index": i,
                    "total": len(strategies),
                    "tagline": strategy.tagline,
                    "description": strategy.description,
                },
            )

        # Create children
        for strategy in strategies:
            child = DialogueNode(
                id=generate_node_id(),
                strategy=strategy,
                messages=[Message.user(cfg.first_message)],
            )
            tree.add_child(root.id, child)
            # Emit node_added for initial strategy branches
            self._emit(
                "node_added",
                {
                    "id": child.id,
                    "parent_id": root.id,
                    "depth": child.depth,
                    "status": child.status.value,
                    "strategy": child.strategy.tagline if child.strategy else None,
                    "user_intent": None,
                    "message_count": len(child.messages),
                },
            )

        log_phase(
            logger,
            "INIT",
            f"Tree initialized with {len(strategies)} branches",
            indent=1,
        )
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
                key=lambda n: scores.get(
                    n.id, AggregatedScore.zero(cfg.prune_threshold)
                ).aggregated_score,
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
                    n.prune_reason = f"score {score.aggregated_score:.1f} < {cfg.prune_threshold}"
                else:
                    n.prune_reason = "scoring failed"

        return survivors

    async def _get_deep_research_context(self) -> str | None:
        """Get deep research context using DeepResearcher."""
        if not self.config.deep_research:
            return None

        report = await self._researcher.research(
            goal=self.config.goal,
            first_message=self.config.first_message,
        )
        self._research_report = report
        return report

    def _track_research_cost(self, cost_usd: float) -> None:
        """Track external research costs (from GPT Researcher)."""
        self._token_tracker.research_cost_usd += cost_usd

    def _track_usage(self, completion: Completion, phase: str) -> None:
        """Track token usage by phase and model."""
        if not completion.usage:
            return

        # Map phase names to TokenTracker attribute names
        phase_map = {
            "strategy": "strategy_generation",
            "intent": "intent_generation",
            "user": "user_simulation",
            "assistant": "assistant_generation",
            "judge": "judging",
        }

        tracker_phase = phase_map.get(phase, phase)
        model = completion.model or self._token_tracker.model_name
        self._token_tracker.add_usage(model, completion.usage, tracker_phase)

    @property
    def tree(self) -> DialogueTree | None:
        """Get the current tree (available after run())."""
        return self._tree
