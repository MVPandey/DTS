"""Trajectory evaluation component for DTS."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.core.dts.aggregator import aggregate_majority_vote
from backend.core.dts.retry import llm_retry
from backend.core.dts.types import AggregatedScore, DialogueNode
from backend.core.dts.utils import format_message_history, log_phase
from backend.core.prompts import prompts
from backend.llm.types import Message
from backend.utils.logging import logger

if TYPE_CHECKING:
    from backend.llm.client import LLM


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
        deep_research_context: str | None = None,
        provider: str | None = None,
        reasoning_enabled: bool = False,
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
            deep_research_context: Optional research context to inform judging.
            provider: Provider preference for OpenRouter (e.g., "Fireworks").
            reasoning_enabled: Enable reasoning tokens for LLM calls.
        """
        self.llm = llm
        self.goal = goal
        self.model = model
        self.judge_temperature = judge_temperature
        self.prune_threshold = prune_threshold
        self._sem = asyncio.Semaphore(max_concurrency)
        self._on_usage = on_usage
        self.deep_research_context = deep_research_context
        self.provider = provider
        self.reasoning_enabled = reasoning_enabled

    def set_research_context(self, context: str | None) -> None:
        """Set or update the deep research context for judging."""
        self.deep_research_context = context

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
                scores_by_id[node.id] = AggregatedScore.zero(self.prune_threshold)
            else:
                agg, critiques = result
                scores_by_id[node.id] = agg
                node.stats.judge_scores = agg.individual_scores
                node.stats.aggregated_score = agg.aggregated_score
                if critiques:
                    node.stats.critiques = critiques

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

        log_phase(
            logger,
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

    async def _judge_single(self, node: DialogueNode) -> tuple[AggregatedScore, dict | None]:
        """Run 3 parallel judges on a single trajectory. Returns (score, critiques)."""
        history_str = format_message_history(node.messages)

        system_prompt, user_prompt = prompts.trajectory_outcome_judge(
            conversation_goal=self.goal,
            conversation_history=history_str,
            deep_research_context=self.deep_research_context,
        )

        # Run 3 judges in parallel
        tasks = [self._call_llm_json(system_prompt, user_prompt) for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores: list[float] = []
        judge_results: list[dict] = []

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Judge failed: {result}")
                scores.append(0.0)
                judge_results.append({})
            elif result and "total_score" in result:
                scores.append(float(result["total_score"]))
                judge_results.append(result)
            else:
                scores.append(0.0)
                judge_results.append({})

        while len(scores) < 3:
            scores.append(0.0)
            judge_results.append({})

        agg = aggregate_majority_vote(scores[:3], pass_threshold=self.prune_threshold)

        # Extract critique from median judge (the one closest to aggregated score)
        critiques = None
        median_score = agg.aggregated_score
        closest_idx = min(range(3), key=lambda i: abs(scores[i] - median_score))
        median_result = judge_results[closest_idx]

        if median_result:
            # Normalize absolute judge output to match comparative format
            critiques = {
                "strengths": [],
                "weaknesses": [],
                "key_moment": median_result.get("key_turning_point"),
                "summary": median_result.get("summary"),
                "biggest_missed_opportunity": median_result.get("biggest_missed_opportunity"),
            }
            # Extract weaknesses from low-scoring criteria
            criteria = median_result.get("criteria", {})
            for name, data in criteria.items():
                if isinstance(data, dict):
                    score = data.get("score", 1.0)
                    rationale = data.get("rationale", "")
                    if score < 0.5 and rationale:
                        critiques["weaknesses"].append(f"{name}: {rationale}")
                    elif score >= 0.8 and rationale:
                        critiques["strengths"].append(f"{name}: {rationale}")

        return agg, critiques

    async def _judge_single_wrapped(self, node: DialogueNode) -> dict[str, AggregatedScore]:
        """Wrapper to return dict format for gather."""
        agg, critiques = await self._judge_single(node)
        node.stats.judge_scores = agg.individual_scores
        node.stats.aggregated_score = agg.aggregated_score
        if critiques:
            node.stats.critiques = critiques
        return {node.id: agg}

    async def _judge_group_comparative(
        self, parent_id: str, group: list[DialogueNode]
    ) -> dict[str, AggregatedScore]:
        """Judge a group of siblings using comparative ranking."""
        log_phase(
            logger,
            "JUDGE",
            f"Ranking {len(group)} siblings (parent: {parent_id[:8]}...)",
            indent=1,
        )

        trajectories = []
        for node in group:
            trajectories.append(
                {
                    "id": node.id,
                    "intent_label": node.user_intent.label if node.user_intent else "unknown",
                    "history": format_message_history(node.messages),
                }
            )

        system_prompt, user_prompt = prompts.comparative_trajectory_judge(
            conversation_goal=self.goal,
            trajectories=trajectories,
            deep_research_context=self.deep_research_context,
        )

        result = await self._call_llm_json(system_prompt, user_prompt)
        scores_by_id: dict[str, AggregatedScore] = {}

        if not result or "ranking" not in result:
            logger.warning(f"Comparative judge failed for {parent_id}, fallback to absolute")
            return await self._fallback_absolute(group)

        ranking = result.get("ranking", [])
        critiques = result.get("critiques", {})

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
            log_phase(
                logger,
                "JUDGE",
                f"Rank {rank}: '{strategy}' [{intent_label}] = {score}/10",
                indent=2,
            )
            log_phase(logger, "JUDGE", f"  Reason: {reason}", indent=2)

            # Log critiques (strengths, weaknesses, key_moment)
            if node_id in critiques:
                critique = critiques[node_id]
                strengths = critique.get("strengths", [])
                weaknesses = critique.get("weaknesses", [])
                key_moment = critique.get("key_moment", "")

                if strengths:
                    log_phase(logger, "JUDGE", f"  Strengths: {strengths}", indent=2)
                if weaknesses:
                    log_phase(logger, "JUDGE", f"  Weaknesses: {weaknesses}", indent=2)
                if key_moment:
                    log_phase(logger, "JUDGE", f"  Key moment: {key_moment}", indent=2)

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

            # Store critiques if available
            if node_id in critiques:
                node.stats.critiques = critiques[node_id]

        # Handle missing nodes
        for node in group:
            if node.id not in scores_by_id:
                scores_by_id[node.id] = AggregatedScore.zero(self.prune_threshold)
                node.stats.judge_scores = [0.0]
                node.stats.aggregated_score = 0.0

        return scores_by_id

    async def _fallback_absolute(self, group: list[DialogueNode]) -> dict[str, AggregatedScore]:
        """Fallback to absolute scoring for a group."""
        tasks = [self._judge_single(node) for node in group]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores_by_id: dict[str, AggregatedScore] = {}
        for node, result in zip(group, results):
            if isinstance(result, Exception):
                agg = AggregatedScore.zero(self.prune_threshold)
                critiques = None
            else:
                agg, critiques = result
            scores_by_id[node.id] = agg
            node.stats.judge_scores = agg.individual_scores
            node.stats.aggregated_score = agg.aggregated_score
            if critiques:
                node.stats.critiques = critiques

        return scores_by_id

    async def _call_llm_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """Make an LLM call expecting JSON output with retry."""
        async with self._sem:
            return await self._call_llm_json_inner(system_prompt, user_prompt)

    @llm_retry(max_attempts=3)
    async def _call_llm_json_inner(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, Any] | None:
        """Inner LLM call with retry logic."""
        messages = [
            Message.system(system_prompt),
            Message.user(user_prompt),
        ]
        completion = await self.llm.complete(
            messages,
            model=self.model,
            temperature=self.judge_temperature,
            structured_output=True,
            provider=self.provider,
            reasoning_enabled=self.reasoning_enabled,
        )
        if self._on_usage:
            self._on_usage(completion, "judge")
        return completion.data
