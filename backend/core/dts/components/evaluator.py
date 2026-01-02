"""Trajectory evaluation component for DTS."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from backend.core.dts.aggregator import aggregate_majority_vote
from backend.core.dts.types import AggregatedScore, DialogueNode
from backend.core.prompts import prompts
from backend.llm.types import Message

if TYPE_CHECKING:
    from backend.llm.client import LLM

logger = logging.getLogger(__name__)


def _log(phase: str, message: str, indent: int = 0) -> None:
    """Print a formatted log message."""
    prefix = "  " * indent
    print(f"[DTS:{phase}] {prefix}{message}")


class TrajectoryEvaluator:
    """
    Evaluates conversation trajectories using LLM judges.

    Supports two scoring modes:
    - Absolute: 3 independent judges score each trajectory (0-10)
    - Comparative: Sibling trajectories are force-ranked against each other
    """

    def __init__(
        self,
        llm: LLM,
        goal: str,
        model: str | None = None,
        judge_temperature: float = 0.3,
        prune_threshold: float = 6.5,
        max_concurrency: int = 16,
        on_usage: Callable[[Any, str], None] | None = None,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            llm: LLM client for judge calls.
            goal: Conversation goal for evaluation context.
            model: Model to use for judging.
            judge_temperature: Temperature for judge calls (lower = more deterministic).
            prune_threshold: Score threshold for pass/fail determination.
            max_concurrency: Maximum concurrent LLM calls.
            on_usage: Callback for token usage tracking (completion, phase).
        """
        self.llm = llm
        self.goal = goal
        self.model = model
        self.judge_temperature = judge_temperature
        self.prune_threshold = prune_threshold
        self._sem = asyncio.Semaphore(max_concurrency)
        self._on_usage = on_usage

    async def evaluate_absolute(
        self,
        nodes: list[DialogueNode],
    ) -> dict[str, AggregatedScore]:
        """
        Score nodes with 3 independent judges each.

        Each trajectory is evaluated in isolation. Scores are aggregated
        via median voting.
        """
        tasks = [self._judge_single(node) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores_by_id: dict[str, AggregatedScore] = {}

        for node, result in zip(nodes, results):
            if isinstance(result, Exception):
                logger.error(f"Error judging node {node.id}: {result}")
                scores_by_id[node.id] = self._error_score()
            else:
                scores_by_id[node.id] = result
                node.stats.judge_scores = result.individual_scores
                node.stats.aggregated_score = result.aggregated_score

        return scores_by_id

    async def evaluate_comparative(
        self,
        nodes: list[DialogueNode],
    ) -> dict[str, AggregatedScore]:
        """
        Score nodes using comparative ranking within sibling groups.

        Nodes with the same parent are force-ranked against each other,
        producing more discriminative scores than absolute judging.
        """
        if len(nodes) <= 1:
            return await self.evaluate_absolute(nodes)

        # Group nodes by parent (siblings compete)
        groups: dict[str, list[DialogueNode]] = {}
        for node in nodes:
            parent_id = node.parent_id or "root"
            if parent_id not in groups:
                groups[parent_id] = []
            groups[parent_id].append(node)

        # Separate single-node groups from multi-node groups
        single_nodes: list[DialogueNode] = []
        multi_groups: list[tuple[str, list[DialogueNode]]] = []

        for parent_id, group in groups.items():
            if len(group) == 1:
                single_nodes.append(group[0])
            else:
                multi_groups.append((parent_id, group))

        # Execute all judging in parallel
        tasks = []
        for node in single_nodes:
            tasks.append(self._judge_single_wrapped(node))
        for parent_id, group in multi_groups:
            tasks.append(self._judge_group_comparative(parent_id, group))

        _log(
            "JUDGE",
            f"Judging {len(single_nodes)} single + {len(multi_groups)} groups in parallel...",
            indent=1,
        )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        scores_by_id: dict[str, AggregatedScore] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Judge task failed: {result}")
                continue
            scores_by_id.update(result)

        return scores_by_id

    async def _judge_single(self, node: DialogueNode) -> AggregatedScore:
        """Run 3 parallel judges on a single trajectory."""
        history_str = self._format_history(node.messages)

        prompt = prompts.trajectory_outcome_judge(
            conversation_goal=self.goal,
            conversation_history=history_str,
        )

        # Run 3 judges in parallel
        tasks = [self._call_llm_json(prompt) for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores: list[float] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Judge failed: {result}")
                scores.append(0.0)
            elif result and "total_score" in result:
                scores.append(float(result["total_score"]))
            else:
                scores.append(0.0)

        while len(scores) < 3:
            scores.append(0.0)

        return aggregate_majority_vote(scores[:3], pass_threshold=self.prune_threshold)

    async def _judge_single_wrapped(
        self, node: DialogueNode
    ) -> dict[str, AggregatedScore]:
        """Wrapper to return dict format for gather."""
        result = await self._judge_single(node)
        node.stats.judge_scores = result.individual_scores
        node.stats.aggregated_score = result.aggregated_score
        return {node.id: result}

    async def _judge_group_comparative(
        self, parent_id: str, group: list[DialogueNode]
    ) -> dict[str, AggregatedScore]:
        """Judge a group of siblings using comparative ranking."""
        _log(
            "JUDGE",
            f"Ranking {len(group)} siblings (parent: {parent_id[:8]}...)",
            indent=1,
        )

        trajectories = []
        for node in group:
            trajectories.append(
                {
                    "id": node.id,
                    "intent_label": node.user_intent.label
                    if node.user_intent
                    else "unknown",
                    "history": self._format_history(node.messages),
                }
            )

        prompt = prompts.comparative_trajectory_judge(
            conversation_goal=self.goal,
            trajectories=trajectories,
        )

        result = await self._call_llm_json(prompt)
        scores_by_id: dict[str, AggregatedScore] = {}

        if not result or "ranking" not in result:
            logger.warning(
                f"Comparative judge failed for {parent_id}, fallback to absolute"
            )
            return await self._fallback_absolute(group)

        ranking = result.get("ranking", [])
        critiques = result.get("critiques", {})

        for node_id, critique in critiques.items():
            weaknesses = critique.get("weaknesses", [])
            if weaknesses:
                _log("JUDGE", f"Critiques for {node_id[:8]}: {weaknesses}", indent=2)

        for entry in ranking:
            node_id = entry.get("trajectory_id", "")
            rank = entry.get("rank", 999)
            score = entry.get("score", 0.0)
            reason = entry.get("reason", "")

            node = next((n for n in group if n.id == node_id), None)
            if not node:
                continue

            intent_label = node.user_intent.label if node.user_intent else "?"
            strategy = node.strategy.tagline if node.strategy else "unknown"
            _log(
                "JUDGE",
                f"Rank {rank}: '{strategy}' [{intent_label}] = {score}/10 - {reason}",
                indent=2,
            )

            agg = AggregatedScore(
                individual_scores=[score, score, score],
                aggregated_score=score,
                pass_threshold=self.prune_threshold,
                pass_votes=3 if score >= self.prune_threshold else 0,
                passed=score >= self.prune_threshold,
            )
            scores_by_id[node_id] = agg
            node.stats.judge_scores = [score]
            node.stats.aggregated_score = score

        # Handle missing nodes
        for node in group:
            if node.id not in scores_by_id:
                scores_by_id[node.id] = self._error_score()
                node.stats.judge_scores = [0.0]
                node.stats.aggregated_score = 0.0

        return scores_by_id

    async def _fallback_absolute(
        self, group: list[DialogueNode]
    ) -> dict[str, AggregatedScore]:
        """Fallback to absolute scoring for a group."""
        tasks = [self._judge_single(node) for node in group]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores_by_id: dict[str, AggregatedScore] = {}
        for node, result in zip(group, results):
            if isinstance(result, Exception):
                result = self._error_score()
            scores_by_id[node.id] = result
            node.stats.judge_scores = result.individual_scores
            node.stats.aggregated_score = result.aggregated_score

        return scores_by_id

    def _error_score(self) -> AggregatedScore:
        """Return a zero score for error cases."""
        return AggregatedScore(
            individual_scores=[0.0, 0.0, 0.0],
            aggregated_score=0.0,
            pass_threshold=self.prune_threshold,
            pass_votes=0,
            passed=False,
        )

    def _format_history(self, messages: list[Message]) -> str:
        """Format messages for prompts."""
        lines = []
        for msg in messages:
            role = msg.role.capitalize()
            content = msg.content or ""
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    async def _call_llm_json(self, prompt: str) -> dict[str, Any] | None:
        """Make an LLM call expecting JSON output."""
        async with self._sem:
            try:
                completion = await self.llm.complete(
                    [Message.user(prompt)],
                    model=self.model,
                    temperature=self.judge_temperature,
                    structured_output=True,
                )
                if self._on_usage:
                    self._on_usage(completion, "judge")
                return completion.data
            except Exception as e:
                logger.error(f"JSON LLM call failed: {e}")
                return None
