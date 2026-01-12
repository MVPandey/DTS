# -----------------------------------------------------------------------------
# Type Alias
# -----------------------------------------------------------------------------
PromptPair = tuple[str, str]  # (system_content, user_content)


# -----------------------------------------------------------------------------
# Class: PromptService
# -----------------------------------------------------------------------------
class PromptService:
    """
    Service for managing prompt templates.

    All prompts return a tuple of (system_content, user_content) for proper
    message role separation.
    """

    # -------------------------------------------------------------------------
    # Strategy Generation
    # -------------------------------------------------------------------------

    def conversation_tree_generator(
        self,
        num_nodes: int,
        conversation_goal: str,
        conversation_context: str,
        deep_research_context: str | None = None,
    ) -> PromptPair:
        """Generate diverse conversation strategies."""
        system = """You are a strategic conversation planner. Generate diverse, orthogonal approaches for achieving conversation goals. Each strategy should explore a genuinely different dimension of the problem space.

You must output valid JSON only. No markdown code fences, no preamble."""

        research_section = ""
        if deep_research_context:
            research_section = f"""

Research context:
{deep_research_context}
"""

        user = f"""Goal: {conversation_goal}

User's message: {conversation_context}
{research_section}
Generate exactly {num_nodes} distinct conversation strategies.

Requirements:
- Each strategy must be orthogonal (different dimension, not a variation)
- Strategies should be generative (can expand into rich sub-branches)
- Taglines should be 2-5 words

Output format:
{{
  "goal": "One-sentence restatement of the goal",
  "nodes": {{
    "Strategy tagline": "One sentence explaining the strategic angle",
    "Another tagline": "One sentence explanation"
  }}
}}"""

        return system, user

    # -------------------------------------------------------------------------
    # Intent Generation
    # -------------------------------------------------------------------------

    def user_intent_generator(
        self,
        num_intents: int,
        conversation_goal: str,
        conversation_history: str,
    ) -> PromptPair:
        """Generate diverse user response intents."""
        system = """You analyze conversations to generate diverse, plausible user response intents. Create orthogonal user behaviors that stress-test different conversation branches.

You must output valid JSON only. No markdown code fences, no preamble."""

        user = f"""Goal: {conversation_goal}

Conversation history:
{conversation_history}

Generate exactly {num_intents} distinct user response INTENTS (behavioral directions, not actual responses).

Requirements:
- Each intent must be orthogonal (genuinely different reaction)
- Include at least one "difficult" intent (resistant, confused, challenging)
- Include at least one "cooperative" intent (engaged, accepting)

Emotional tones: engaged, resistant, confused, skeptical, enthusiastic, deflecting, anxious, neutral
Cognitive stances: accepting, questioning, challenging, exploring, withdrawing

Output format:
{{
  "intents": [
    {{
      "id": "short_id",
      "label": "2-4 word label",
      "description": "One sentence describing how the user will respond",
      "emotional_tone": "one of the tones above",
      "cognitive_stance": "one of the stances above"
    }}
  ]
}}"""

        return system, user

    # -------------------------------------------------------------------------
    # User Simulation
    # -------------------------------------------------------------------------

    def user_simulation(
        self,
        conversation_goal: str,
        user_intent: dict | None = None,
    ) -> PromptPair:
        """
        Generate a simulated user response.

        Note: Conversation history is passed as separate messages, not embedded here.
        """
        intent_section = ""
        if user_intent:
            intent_section = f"""
You MUST embody this specific intent:
- Label: {user_intent.get("label", "")}
- Description: {user_intent.get("description", "")}
- Emotional tone: {user_intent.get("emotional_tone", "")}
- Cognitive stance: {user_intent.get("cognitive_stance", "")}
"""

        system = f"""You are simulating a user in a conversation. Respond authentically as they would - not as an idealized or overly cooperative version.

Goal context: {conversation_goal}
{intent_section}
Guidelines:
- Match the user's established voice and communication style
- Real users push back, get confused, change minds, and have emotional reactions
- Balance goal-direction with natural human behavior
- Do NOT fabricate results, data, or context not established in the conversation

Output the next user message only. No meta-commentary, no JSON. Just the raw message.

CRITICAL: Your response must be non-empty. Even resistant users say something."""

        user = "Continue the conversation as the user. Generate the next user message."

        return system, user

    # -------------------------------------------------------------------------
    # Assistant Continuation
    # -------------------------------------------------------------------------

    def assistant_continuation(
        self,
        conversation_goal: str,
        strategy_tagline: str,
        strategy_description: str,
    ) -> PromptPair:
        """
        Generate assistant's next response following a strategy.

        Note: Conversation history is passed as separate messages.
        """
        system = f"""You are the assistant continuing a conversation. Follow the given strategy naturally - the user should experience coherent conversation, not a strategy being deployed.

Goal: {conversation_goal}

Strategy to follow:
- {strategy_tagline}: {strategy_description}

Guidelines:
- The strategy shapes your approach, not your exact words
- Respond to what the user actually said before advancing the strategy
- Match the user's energy and register
- If the strategy conflicts with the conversation's needs, prioritize the conversation

Output the next assistant message only. No meta-commentary, no JSON."""

        user = "Continue the conversation as the assistant. Generate your next response."

        return system, user

    # -------------------------------------------------------------------------
    # Rephrase with Intent
    # -------------------------------------------------------------------------

    def rephrase_with_intent(
        self,
        original_message: str,
        intent_label: str,
        intent_description: str,
        emotional_tone: str,
        cognitive_stance: str,
    ) -> PromptPair:
        """Rephrase a message to incorporate a specific intent."""
        system = """You rephrase messages to incorporate specific emotional tones and cognitive stances while preserving core meaning.

Output ONLY the rephrased message. No explanation, no quotes, no preamble."""

        user = f"""Original message: {original_message}

Rephrase to incorporate:
- Intent: {intent_label} - {intent_description}
- Emotional tone: {emotional_tone}
- Cognitive stance: {cognitive_stance}

Keep roughly the same length. Make it sound natural."""

        return system, user

    # -------------------------------------------------------------------------
    # Trajectory Outcome Judge
    # -------------------------------------------------------------------------

    def trajectory_outcome_judge(
        self,
        conversation_goal: str,
        conversation_history: str,
        deep_research_context: str | None = None,
    ) -> PromptPair:
        """Evaluate a conversation trajectory."""
        system = """You are an EXACTING evaluator of conversation trajectories. Find flaws, identify missed opportunities, and score HARSHLY. Most conversations are mediocre - surface that reality.

Calibration:
- 7/10 = genuinely good conversation
- 8+ = rare, exceptional execution
- 10/10 = flawless, almost never appropriate

You must output valid JSON only. No markdown code fences, no preamble."""

        research_section = ""
        if deep_research_context:
            research_section = f"""

Research context (assess whether choices were sound):
{deep_research_context}
"""

        user = f"""Goal: {conversation_goal}

Conversation:
{conversation_history}
{research_section}
Evaluate against these criteria (0.0-1.0 each, find something to critique):

1. goal_achieved: Was the goal FULLY achieved?
2. user_need_addressed: Was the UNDERLYING need met?
3. forward_progress: Did EVERY turn move forward?
4. user_engagement_maintained: Did engagement INCREASE?
5. rapport_preserved: Was rapport STRENGTHENED?
6. appropriate_resolution: Was the ending OPTIMAL?
7. actionable_outcome: Are next steps CONCRETE?
8. no_harm_done: ANY risk of harm or confusion?
9. efficient_path: Was this the SHORTEST viable path?
10. user_better_off: Is improvement SIGNIFICANT?

Output format:
{{
  "criteria": {{
    "goal_achieved": {{"score": 0.0-1.0, "rationale": "what was lacking"}},
    ...
  }},
  "total_score": sum of all scores (0-10),
  "confidence": "low|medium|high",
  "summary": "One sentence critique",
  "biggest_missed_opportunity": "What could have made this better"
}}

VERIFY: Your total should typically be 4-7. Above 8 requires exceptional justification."""

        return system, user

    # -------------------------------------------------------------------------
    # Branch Selection Judge
    # -------------------------------------------------------------------------

    def branch_selection_judge(
        self,
        conversation_goal: str,
        conversation_context: str,
        branch_tagline: str,
        branch_description: str,
    ) -> PromptPair:
        """Evaluate a branch selection decision."""
        system = """You evaluate conversation branch choices. Score how promising a direction is BEFORE it's explored - like evaluating a chess move based on position, not outcome.

You must output valid JSON only. No markdown code fences, no preamble."""

        user = f"""Goal: {conversation_goal}

Context:
{conversation_context}

Branch selected: {branch_tagline}
Description: {branch_description}

Evaluate against these criteria (0.0, 0.5, or 1.0):

1. goal_aligned: Advances toward the goal?
2. contextually_appropriate: Matches user's expertise level?
3. emotionally_attuned: Respects user's emotional state?
4. well_timed: Right moment for this move?
5. builds_on_history: Connects to what's been discussed?
6. information_generating: Will reveal useful information?
7. not_redundant: Explores new territory?
8. appropriately_scoped: Neither too narrow nor too broad?
9. actionable: Can lead to concrete next steps?
10. low_risk: Unlikely to damage rapport?

Output format:
{{
  "criteria": {{
    "goal_aligned": {{"score": 0|0.5|1, "rationale": "one sentence"}},
    ...
  }},
  "total_score": sum (0-10),
  "confidence": "low|medium|high",
  "summary": "One sentence assessment"
}}"""

        return system, user

    # -------------------------------------------------------------------------
    # Comparative Trajectory Judge
    # -------------------------------------------------------------------------

    def comparative_trajectory_judge(
        self,
        conversation_goal: str,
        trajectories: list[dict],
        deep_research_context: str | None = None,
    ) -> PromptPair:
        """Compare and rank multiple trajectories."""
        system = """You compare conversation trajectories and force-rank them. Find flaws in ALL trajectories and assign scores that CREATE SEPARATION.

Scoring scale:
- Rank 1 = 7.5 (best of set, still has flaws)
- Rank 2 = 6.0
- Rank 3 = 4.5
- Each subsequent rank: subtract 1.5

Only Rank 1 scores 8.5+ if TRULY exceptional. 9+ requires flawless execution.

You must output valid JSON only. No markdown code fences, no preamble."""

        # Format trajectories
        traj_text = ""
        for t in trajectories:
            traj_text += (
                f"\n--- Trajectory {t['id']} (intent: {t.get('intent_label', 'unknown')}) ---\n"
            )
            traj_text += t["history"]
            traj_text += "\n"

        research_section = ""
        if deep_research_context:
            research_section = f"""

Research context:
{deep_research_context}
"""

        user = f"""Goal: {conversation_goal}
{research_section}
Trajectories to compare:
{traj_text}

For each trajectory, find 2-3 specific weaknesses. Then force-rank (NO TIES).

Output format:
{{
  "critiques": {{
    "trajectory_id": {{
      "weaknesses": ["specific weakness 1", "specific weakness 2"],
      "strengths": ["specific strength"],
      "key_moment": "moment that most affected quality"
    }}
  }},
  "ranking": [
    {{
      "rank": 1,
      "trajectory_id": "id",
      "score": 7.5,
      "reason": "Why ranked #1"
    }},
    ...
  ],
  "ranking_confidence": "low|medium|high"
}}

VERIFY: Rank 1 <= 8.0 unless exceptional. Gap between ranks >= 1.0 point."""

        return system, user


prompts = PromptService()
