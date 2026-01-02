import re
from typing import Any


class PromptService:
    """
    Service for managing and rendering prompt templates.

    Usage:
        prompts = PromptService()
        prompt = prompts.conversation_tree_generator(
            num_nodes=5,
            conversation_context="User wants to learn Python",
            deep_research_context=None
        )
    """

    USER_INTENT_GENERATOR = """
<system>
You are analyzing a conversation to generate diverse, plausible user response intents. Your goal is to create orthogonal user behaviors that will stress-test different conversation branches.
</system>

<task>
Given a conversation history and the assistant's last response, generate exactly {{num_intents}} distinct user response INTENTS. These are not the responses themselves—they are behavioral directions the user might take.

Each intent must be:
- **Orthogonal**: Represents a genuinely different user reaction, not a variation of another intent
- **Plausible**: A real user could reasonably respond this way given the context
- **Revealing**: Will expose whether the assistant's approach is robust or fragile
</task>

<input_context>
<goal>
{{conversation_goal}}
</goal>

<conversation_history>
{{conversation_history}}
</conversation_history>
</input_context>

<output_format>
Respond with valid JSON only. No markdown code fences, no preamble.

{
  "intents": [
    {
      "id": "cooperative",
      "label": "Short 2-4 word label",
      "description": "One sentence describing how the user will respond and why",
      "emotional_tone": "engaged|resistant|confused|skeptical|enthusiastic|deflecting|anxious|neutral",
      "cognitive_stance": "accepting|questioning|challenging|exploring|withdrawing"
    }
  ]
}

Requirements:
- Generate exactly {{num_intents}} intents
- Intents should span the realistic range of user reactions
- At least one intent should be "difficult" (resistant, confused, or challenging)
- At least one intent should be "cooperative" (engaged, accepting)
- Labels should be unique and descriptive
</output_format>

<calibration_examples>
<example type="technical">
<context>
Goal: Debug a memory leak
Last assistant message: "Try running memory_profiler on your main loop"
</context>
<intents>
[
  {"id": "compliant", "label": "Tries the tool", "description": "User runs the profiler and shares output", "emotional_tone": "engaged", "cognitive_stance": "accepting"},
  {"id": "resistant", "label": "Rejects dependency", "description": "User pushes back on adding a new dependency", "emotional_tone": "resistant", "cognitive_stance": "challenging"},
  {"id": "confused", "label": "Needs clarification", "description": "User doesn't know how to install or run the tool", "emotional_tone": "confused", "cognitive_stance": "questioning"},
  {"id": "tangent", "label": "Reveals more context", "description": "User mentions this only happens in production, adding complexity", "emotional_tone": "neutral", "cognitive_stance": "exploring"}
]
</intents>
</example>

<example type="emotional">
<context>
Goal: Process feelings about breakup
Last assistant message: "That sounds really painful. How are you holding up?"
</context>
<intents>
[
  {"id": "opens_up", "label": "Shares vulnerability", "description": "User opens up about the raw pain and specific memories", "emotional_tone": "engaged", "cognitive_stance": "accepting"},
  {"id": "deflects", "label": "Minimizes feelings", "description": "User deflects with 'I'm fine' or rationalizes the breakup", "emotional_tone": "deflecting", "cognitive_stance": "withdrawing"},
  {"id": "redirects", "label": "Asks for advice", "description": "User asks what they should do or how to feel better", "emotional_tone": "anxious", "cognitive_stance": "questioning"},
  {"id": "challenges", "label": "Tests empathy", "description": "User pushes back: 'You can't really understand this'", "emotional_tone": "skeptical", "cognitive_stance": "challenging"}
]
</intents>
</example>
</calibration_examples>

<instructions>
1. Analyze the conversation history to understand the current state
2. Consider what the assistant just said/asked and how a real user might react
3. Generate {{num_intents}} orthogonal intents spanning cooperative to difficult
4. Ensure intents would actually reveal something about the quality of different assistant strategies
5. Make at least one intent challenging—this stress-tests the branches
</instructions>
""".strip()

    CONVERSATION_TREE_GENERATOR = """
<system>
You are a strategic conversation planner. Given a conversational end-goal and context, generate initial branching nodes for exploring the solution space—similar to root moves in Monte Carlo Tree Search.
</system>

<task>
Produce exactly {{num_nodes}} distinct, high-potential starting approaches that represent fundamentally different paths toward the goal.

Each node must be:
- **Orthogonal**: Explores a genuinely different dimension of the problem space, not a variation of another node
- **Generative**: Naturally expands into rich sub-branches for further exploration
- **Evaluable**: Allows assessment of whether pursuing this branch is productive
</task>

<input_context>
<goal>
{{conversation_goal}}
</goal>
<conversation_history>
{{conversation_context}}
</conversation_history>

{{#if deep_research_context}}
<deep_research>
The following research provides additional domain context relevant to this conversation:

{{deep_research_context}}
</deep_research>
{{/if}}
</input_context>

<output_format>
Respond with valid JSON only. No markdown code fences, no preamble, no explanation outside the JSON.

{
  "goal": "A clear, one-sentence restatement of the conversational end-goal",
  "nodes": {
    "Short tagline for branch 1": "One sentence explaining the strategic angle and why this path is promising",
    "Short tagline for branch 2": "One sentence explaining the strategic angle and why this path is promising"
  },
  "coverage_rationale": "Brief explanation of why these nodes collectively span the solution space well"
}

Requirements:
- "nodes" must contain exactly {{num_nodes}} key-value pairs
- Taglines should be 2-5 words, suitable as conversation pivot points
- Descriptions should be actionable and specific to the context provided
</output_format>

<examples>
<example type="technical">
<input>
Conversation context: User is researching optimizer improvements and has mentioned interest in the Muon optimizer. They have a background in ML systems.
Num nodes: 6
</input>
<output>
{
  "goal": "Discover viable improvements to the Muon optimizer algorithm",
  "nodes": {
    "Theoretical foundations": "Examine mathematical assumptions in orthogonalization and Newton-Schulz iterations for weaknesses or generalizations",
    "Compute efficiency": "Target FLOP and memory bottlenecks through kernel fusion, reduced-precision variants, or algorithmic shortcuts",
    "Hyperparameter sensitivity": "Investigate fragile settings and whether adaptive or scheduled schemes could improve robustness",
    "Scaling behavior": "Probe performance degradation at extremes of model size, batch size, and sequence length",
    "Hybrid approaches": "Explore combinations with other optimizers via warmup schedules, switching conditions, or gradient ensembling",
    "Domain transfer": "Test failure modes on non-LLM architectures including vision, reinforcement learning, and sparse models"
  },
  "coverage_rationale": "Nodes span theory vs. practice, speed vs. quality, and core algorithm vs. integration—covering the main axes of optimizer research while respecting the user's systems background."
}
</output>
</example>

<example type="emotional_support">
<input>
Conversation context: User has expressed feeling persistently low and unmotivated for the past few weeks. They mentioned work stress but haven't elaborated. Tone is somewhat withdrawn.
Num nodes: 5
</input>
<output>
{
  "goal": "Support user toward improved emotional state and restored sense of agency",
  "nodes": {
    "Validate and witness": "Create psychological safety through acknowledgment before attempting any intervention or advice",
    "Somatic factors": "Gently explore physical dimensions like sleep, movement, and nutrition as accessible levers",
    "Temporal pattern": "Understand duration, triggers, and fluctuation to distinguish acute situational response from chronic state",
    "Work stress unpacking": "Follow the thread they opened about work to understand specific pressures and whether they feel actionable",
    "Micro-agency restoration": "Identify one tiny, concrete action to rebuild sense of control without overwhelming"
  },
  "coverage_rationale": "Balances validation-first with action-oriented paths, spans internal and external factors, and follows up on the work thread they introduced while respecting withdrawn tone."
}
</output>
</example>
</examples>

<instructions>
1. Analyze the conversation context to understand the implicit and explicit goals
2. Consider the user's apparent expertise level, emotional state, and what they've already explored
3. Generate {{num_nodes}} orthogonal branches that maximize coverage of promising directions
4. Ensure each branch could realistically advance the conversation toward the goal
5. If deep research context is provided, incorporate relevant insights but don't let it dominate over conversation-specific signals
</instructions>
"""

    BRANCH_SELECTION_JUDGE_PROMPT = """
    <system>
    You are an expert evaluator assessing the quality of a conversational branch choice. Your role is to score how promising a particular conversation direction is BEFORE it's explored—like evaluating a chess move based on position, not outcome.
    </system>

    <task>
    Evaluate the selected branch against 10 binary criteria. Award 1 point for clearly met, any range of numbers between 0 and 1 for partially met, and 0 for not met. You are one of several judges whose scores will be aggregated, so prioritize your honest independent assessment.

    You are judging the DECISION to take this branch, not its eventual outcome. A good branch choice can still lead to a bad outcome, and vice versa. Focus on: given what we know right now, is this a smart direction to explore?
    </task>

    <input_context>
    <goal>
    {{conversation_goal}}
    </goal>

    <conversation_history>
    {{conversation_context}}
    </conversation_history>

    <branch_selected>
    {{branch_tagline}}: {{branch_description}}
    </branch_selected>
    </input_context>

    <rubric>
    For each criterion, award:
    - 1.0: Clearly met
    - 0.5: Partially met or uncertain
    - 0.0: Not met

    <criteria>
    1. **goal_aligned**: Does this branch directly advance toward the stated goal?
    2. **contextually_appropriate**: Does this branch match the user's apparent expertise level and domain?
    3. **emotionally_attuned**: Does this branch respect the user's emotional state and tone?
    4. **well_timed**: Is this the right moment in the conversation for this type of move?
    5. **builds_on_history**: Does this branch connect to or leverage what's already been discussed?
    6. **information_generating**: Will this branch likely reveal useful information about the right path forward?
    7. **not_redundant**: Does this branch explore new territory rather than retreading covered ground?
    8. **appropriately_scoped**: Is this branch neither too narrow (dead end) nor too broad (unfocused)?
    9. **actionable**: Can this branch plausibly lead to concrete next steps or insights?
    10. **low_risk**: Is this branch unlikely to damage rapport, trust, or conversation momentum?
    </criteria>
    </rubric>

    <output_format>
    Respond with valid JSON only. No markdown code fences, no preamble.

    {
    "criteria": {
        "goal_aligned": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "contextually_appropriate": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "emotionally_attuned": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "well_timed": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "builds_on_history": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "information_generating": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "not_redundant": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "appropriately_scoped": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "actionable": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        },
        "low_risk": {
        "score": <0 | 0.01 - 0.99 | 1>,
        "rationale": "<One sentence justification>"
        }
    },
    "total_score": <sum of all criteria, 0-10>,
    "confidence": "<low|medium|high>",
    "summary": "<One sentence overall assessment of this branch choice>"
    }
    </output_format>

    <calibration_examples>
    <example type="strong_selection">
    <scenario>
    Goal: Help user debug a memory leak in their Python application
    Context: User shared stack trace, mentioned they're experienced with Python but new to memory profiling
    Branch: "Profiling tooling" - Introduce memory_profiler and tracemalloc as diagnostic starting points
    </scenario>
    <scores>
    goal_aligned: 1 (directly addresses the debugging goal)
    contextually_appropriate: 1 (matches Python expertise, fills profiling gap)
    emotionally_attuned: 1 (neutral technical context, no emotional mismatch)
    well_timed: 1 (profiling is the logical first diagnostic step)
    builds_on_history: 1 (responds to the stack trace they shared)
    information_generating: 1 (profiler output will narrow the search space)
    not_redundant: 1 (profiling tools not yet discussed)
    appropriately_scoped: 1 (specific tools with clear application)
    actionable: 1 (user can immediately run these)
    low_risk: 1 (no rapport risk in technical suggestion)
    Total: 10/10
    </scores>
    </example>

    <example type="weak_selection">
    <scenario>
    Goal: Help user process feelings about a recent breakup
    Context: User just shared that the breakup happened yesterday, expressing raw grief
    Branch: "Growth reframe" - Discuss how this could be an opportunity for personal development
    </scenario>
    <scores>
    goal_aligned: 0.5 (growth framing is part of processing, but not the immediate need)
    contextually_appropriate: 0 (misreads what's needed in acute grief)
    emotionally_attuned: 0 (dismisses current emotional state)
    well_timed: 0 (premature—validation must precede reframing)
    builds_on_history: 0.5 (acknowledges breakup but ignores emotional signals)
    information_generating: 0 (user likely to shut down, learning nothing)
    not_redundant: 1 (hasn't been discussed yet)
    appropriately_scoped: 0.5 (too abstract for the moment)
    actionable: 0 (user not in a state to act on growth framing)
    low_risk: 0 (high risk of damaging trust)
    Total: 2.5/10
    </scores>
    </example>
    </calibration_examples>

    <instructions>
    1. Read the goal and conversation context carefully
    2. Evaluate each criterion independently—don't let one judgment contaminate others
    3. Judge the BRANCH CHOICE, not hypothetical outcomes
    4. Use any number between 0 and 1 to score partially met or uncertain, but not 0 or 1
    5. Sum the scores accurately
    6. Set confidence based on how much context you have to make these judgments
    </instructions>
    """

    TRAJECTORY_OUTCOME_JUDGE_PROMPT = """
<system>
You are an EXACTING evaluator assessing conversational trajectories. Your role is to find flaws, identify missed opportunities, and score HARSHLY. Most conversations are mediocre—your job is to surface that reality, not inflate scores.

You are calibrated to be a tough grader. A score of 7/10 represents a genuinely good conversation. Scores above 8 are rare and reserved for exceptional execution. A perfect 10 requires flawless performance with zero critiques possible.
</system>

<task>
Evaluate this trajectory against 10 criteria. You MUST be critical. For each criterion, actively look for what went wrong or could have been better.

Scoring philosophy:
- 1.0: Flawless. Literally nothing could be improved. Almost never appropriate.
- 0.8-0.9: Excellent. Minor nitpicks only.
- 0.6-0.7: Good. Solid execution but clear room for improvement.
- 0.4-0.5: Adequate. Gets the job done but notable weaknesses.
- 0.2-0.3: Poor. Significant problems.
- 0.0-0.1: Failed. Did not meet this criterion.

Your total score should typically land between 4-7. Scores above 8 require exceptional justification. Scores of 9+ should be extremely rare.
</task>

<input_context>
<goal>
{{conversation_goal}}
</goal>

<conversation_history>
{{conversation_history}}
</conversation_history>
</input_context>

<rubric>
For each criterion, find something to critique. If you cannot find ANY flaw, explain why this is truly flawless (this should be rare).

<criteria>
1. **goal_achieved**: Was the stated goal FULLY achieved, not just partially addressed?
2. **user_need_addressed**: Was the UNDERLYING need met, not just the surface request?
3. **forward_progress**: Did EVERY turn move forward, or were there stalls, circles, or wasted exchanges?
4. **user_engagement_maintained**: Did engagement INCREASE or just not decrease? Look for signs of waning interest.
5. **rapport_preserved**: Was rapport STRENGTHENED, not just maintained? Any moments of friction?
6. **appropriate_resolution**: Was the ending OPTIMAL, not just acceptable? Could it have ended better?
7. **actionable_outcome**: Are next steps CONCRETE and IMMEDIATELY actionable, not vague?
8. **no_harm_done**: Was there ANY risk of harm, confusion, or misinformation, even minor?
9. **efficient_path**: Was this the SHORTEST viable path? Any unnecessary tangents?
10. **user_better_off**: Is the improvement SIGNIFICANT and DEMONSTRABLE, not just marginal?
</criteria>
</rubric>

<output_format>
Respond with valid JSON only. No markdown code fences, no preamble.

{
  "criteria": {
    "goal_achieved": {
      "score": <0.0-1.0>,
      "rationale": "<What was lacking or could be improved, OR why this is truly flawless>"
    },
    "user_need_addressed": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "forward_progress": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "user_engagement_maintained": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "rapport_preserved": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "appropriate_resolution": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "actionable_outcome": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "no_harm_done": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "efficient_path": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    },
    "user_better_off": {
      "score": <0.0-1.0>,
      "rationale": "<Critique or flawless justification>"
    }
  },
  "total_score": <sum of all criteria, 0-10>,
  "confidence": "<low|medium|high>",
  "summary": "<One sentence critique of the trajectory>",
  "key_turning_point": "<The moment that most determined success or failure, or null if none>",
  "biggest_missed_opportunity": "<What could have made this significantly better>"
}
</output_format>

<calibration_examples>
<example type="good_not_great">
<scenario>
Goal: Help user debug a memory leak in their Python application
Conversation: Introduced profiling tools → User ran memory_profiler → Identified leak in cache dictionary → Suggested LRU cache → User implemented fix → Confirmed resolved
</scenario>
<expected_scores>
goal_achieved: 0.9 (solved but didn't verify no other leaks exist)
user_need_addressed: 0.8 (fixed this leak but didn't teach debugging skills)
forward_progress: 0.85 (mostly linear but could have asked about cache size upfront)
user_engagement_maintained: 0.7 (user followed instructions but didn't show enthusiasm)
rapport_preserved: 0.7 (functional interaction, no warmth built)
appropriate_resolution: 0.75 (ended at fix, didn't discuss prevention)
actionable_outcome: 0.9 (clear fix implemented)
no_harm_done: 0.95 (no issues, minor: didn't warn about LRU memory tradeoffs)
efficient_path: 0.7 (could have asked about data patterns earlier)
user_better_off: 0.8 (problem solved but missed teaching moment)
Total: 8.1/10 - A good debugging session, but mechanical rather than educational
</expected_scores>
</example>

<example type="mediocre">
<scenario>
Goal: Help user process feelings about a recent breakup
Conversation: Opened with validation → User shared context → Explored grief → User mentioned not sleeping → Pivoted to sleep hygiene → User thanked and left
</scenario>
<expected_scores>
goal_achieved: 0.4 (pivoted away from emotional processing to practical advice)
user_need_addressed: 0.3 (user needed to be heard, got sleep tips instead)
forward_progress: 0.5 (moved but in wrong direction)
user_engagement_maintained: 0.5 (user disengaged with polite "thanks")
rapport_preserved: 0.6 (not damaged but not deepened)
appropriate_resolution: 0.3 (premature ending, user fled)
actionable_outcome: 0.6 (sleep tips are actionable but wrong target)
no_harm_done: 0.7 (no direct harm but emotional bypassing)
efficient_path: 0.4 (took a detour from the actual goal)
user_better_off: 0.3 (marginally, but core need unmet)
Total: 4.6/10 - Well-intentioned but missed what the user actually needed
</expected_scores>
</example>

<example type="failed">
<scenario>
Goal: Help user decide between two job offers
Conversation: Asked about priorities → User listed factors → Started comparing → User mentioned resignation anxiety → Spiraled into fears → Never returned to comparison → User said "I'm more confused now"
</scenario>
<expected_scores>
goal_achieved: 0.1 (explicitly failed - user more confused)
user_need_addressed: 0.2 (surfaced anxiety but didn't help with it)
forward_progress: 0.2 (backwards movement)
user_engagement_maintained: 0.4 (engaged in spiral, not productively)
rapport_preserved: 0.5 (user stayed but trust unclear)
appropriate_resolution: 0.1 (no resolution, user fled confused)
actionable_outcome: 0.1 (no action possible from this)
no_harm_done: 0.4 (increased anxiety without resolution)
efficient_path: 0.1 (completely off track)
user_better_off: 0.0 (explicitly worse off)
Total: 2.1/10 - Derailed by anxiety without recovery
</expected_scores>
</example>
</calibration_examples>

<anti_inflation_rules>
BEFORE submitting, verify:
1. Your total score is NOT above 8.0 unless you can articulate why this is exceptional
2. You have found at least ONE critique for MOST criteria
3. You have not given more than TWO scores of 1.0
4. "Good enough" execution scores 0.6-0.7, not 0.8-0.9
5. Polite endings are often escape behaviors—don't score them highly
</anti_inflation_rules>

<instructions>
1. Read the full conversation with a critical eye
2. For EACH criterion, first identify what could be better
3. Only after finding critiques, assign a score that reflects the gap from perfection
4. Verify your total aligns with calibration examples
5. If your total exceeds 8.0, re-examine for missed critiques
</instructions>
""".strip()

    COMPARATIVE_TRAJECTORY_JUDGE = """
<system>
You are an EXACTING evaluator comparing conversation trajectories. Your job is to find flaws, force hard choices between options, and assign scores that CREATE SEPARATION between branches. Even "good" trajectories have weaknesses—find them.

Scoring is HARSH and SPREAD OUT:
- Rank 1 = 7.5 (best of this set, but still has flaws)
- Rank 2 = 6.0 (second best)
- Rank 3 = 4.5 (third)
- Rank 4 = 3.0 (fourth)
- Additional ranks continue down by 1.5

Only if Rank 1 is TRULY exceptional (rare) should it score 8.5+. A score of 9+ requires flawless execution.
</system>

<task>
You are given {{num_trajectories}} conversation trajectories that all started from the same context but diverged. Your job is to:

1. **Critique HARSHLY**: Find 2-3 specific weaknesses for EACH trajectory. No trajectory is above criticism.
2. **Force-rank them**: Assign positions 1 through {{num_trajectories}} (NO TIES)
3. **Spread scores widely**: Use the scoring scale above. Do NOT cluster scores near the top.

You MUST discriminate. The gap between Rank 1 and Rank 2 should be MEANINGFUL, not cosmetic.
</task>

<input_context>
<goal>
{{conversation_goal}}
</goal>

<trajectories>
{{#each trajectories}}
<trajectory id="{{this.id}}" intent="{{this.intent_label}}">
{{this.history}}
</trajectory>
{{/each}}
</trajectories>
</input_context>

<evaluation_criteria>
When comparing, consider:
1. **Goal achievement**: Which trajectory made the most meaningful progress toward the goal?
2. **User handling**: Which best adapted to the user's intent/behavior (cooperative, resistant, confused, etc.)?
3. **Missed opportunities**: Which trajectory left the most value on the table?
4. **Conversation quality**: Which felt most natural and productive?
5. **Robustness**: Which would work best across different user types?
</evaluation_criteria>

<output_format>
Respond with valid JSON only. No markdown code fences, no preamble.

{
  "critiques": {
    "<trajectory_id_1>": {
      "weaknesses": ["Specific weakness 1", "Specific weakness 2"],
      "strengths": ["Specific strength 1"],
      "key_moment": "The moment that most affected this trajectory's quality"
    },
    "<trajectory_id_2>": { ... }
  },
  "ranking": [
    {
      "rank": 1,
      "trajectory_id": "<id of best trajectory>",
      "score": 10,
      "reason": "One sentence explaining why this is ranked #1"
    },
    {
      "rank": 2,
      "trajectory_id": "<id of second best>",
      "score": 8,
      "reason": "One sentence explaining the gap from #1"
    }
  ],
  "ranking_confidence": "<low|medium|high>",
  "discrimination_difficulty": "One sentence on how hard it was to rank these"
}
</output_format>

<calibration_example>
<scenario>
Goal: Help user debug a memory leak
Trajectories:
- A [cooperative]: User ran profiler → found leak → got fix
- B [resistant]: User refused dependency → got stdlib alternative → slower progress
- C [confused]: User didn't understand output → assistant explained → eventually found leak
</scenario>
<expected_ranking>
1. A (score: 7.5) - Direct path but didn't teach debugging skills or verify no other leaks
2. C (score: 6.0) - Educational approach but took longer; user learned something
3. B (score: 4.5) - Adapted to resistance but stdlib path was less thorough
</expected_ranking>
<key_insight>Even the "best" trajectory (A) only scores 7.5—it solved the immediate problem but missed opportunities for deeper value.</key_insight>
</calibration_example>

<anti_inflation_rules>
BEFORE submitting, verify:
1. Rank 1 is NOT scored above 8.0 unless you can articulate exceptional execution
2. The gap between Rank 1 and Rank 2 is at least 1.0 points
3. You have found weaknesses for EVERY trajectory, including Rank 1
4. Scores are SPREAD OUT, not clustered (e.g., NOT 8.5, 8.0, 7.5)
</anti_inflation_rules>

<instructions>
1. Read all trajectories with a CRITICAL eye
2. For EACH trajectory, find specific weaknesses FIRST, then strengths
3. Compare head-to-head on each criterion—which handled it better?
4. Rank based on overall quality. Handling difficult users well is a STRENGTH.
5. Apply scores: Rank 1 = 7.5 (or 8.0-8.5 if exceptional), then subtract 1.5 for each subsequent rank
6. Verify scores are spread out and not inflated
</instructions>
""".strip()

    USER_SIMULATOR_PROMPT = """
<system>
You are simulating the user in a conversation. Your job is to embody this user authentically—responding as they would, not as an idealized or cooperative version of them.
</system>

<task>
Generate the next user message in this conversation. You must balance two tensions:
1. **Goal-directed**: You have an underlying goal you're trying to achieve
2. **Naturalistically human**: You don't always know exactly what you want, you react emotionally, you sometimes resist, digress, or need convincing

Real users are not cooperative assistants. They push back, get confused, change their minds, miss the point, fixate on tangents, and have emotional reactions that interrupt logical flow. Simulate this.
</task>

<input_context>
<goal>
{{conversation_goal}}
</goal>

<conversation_history>
{{conversation_history}}
</conversation_history>

{{#if user_intent}}
<assigned_intent>
You MUST embody this specific user intent in your response:
- **Intent**: {{user_intent_label}}
- **Description**: {{user_intent_description}}
- **Emotional tone**: {{user_intent_tone}}
- **Cognitive stance**: {{user_intent_stance}}

This intent is non-negotiable. Your response must clearly reflect this behavioral direction while remaining natural and plausible.
</assigned_intent>
{{/if}}
</input_context>

<behavioral_guidelines>
<authenticity>
- Match the user's established voice, vocabulary, and communication style from the conversation history
- If no history exists, infer plausible user traits from the goal (technical goals suggest technical users, emotional goals suggest someone in that emotional state)
- Maintain consistent personality across turns—don't suddenly become more articulate or cooperative
</authenticity>

<cognitive_realism>
- You may not fully understand the assistant's response on first read
- You might latch onto one part of a multi-part response and ignore the rest
- You can ask for clarification, repetition, or simpler explanations
- You might misinterpret something and respond to what you thought was said
- You can change your mind mid-conversation as new information lands
</cognitive_realism>

<emotional_realism>
- If the goal involves emotional content, inhabit that emotional state—don't just describe it
- Emotions affect cognition: anxious users spiral, frustrated users get short, grieving users may not want solutions
- You can warm up or cool down toward the assistant based on how the conversation goes
- Defensiveness, skepticism, and resistance are valid responses when appropriate
</emotional_realism>

<engagement_patterns>
When assistant offers ideas or suggestions:
- Sometimes accept readily if it resonates
- Sometimes probe deeper: "What do you mean by that?" or "How would that actually work?"
- Sometimes push back: "I don't think that applies here because..." or "I already tried that"
- Sometimes digress: "That reminds me of..." or "Actually, the bigger issue is..."
- Sometimes express doubt: "I'm not sure..." or "That sounds hard"

When assistant asks questions:
- Sometimes answer directly
- Sometimes answer partially or tangentially
- Sometimes deflect: "I don't know" or "That's not really the issue"
- Sometimes answer with a question: "Why do you ask?" or "What do you think?"
</engagement_patterns>

<goal_awareness>
Your awareness of the goal varies by context:
- **Explicit task** (e.g., "debug this code"): You know what you want, you're evaluating if the assistant is helping
- **Exploratory** (e.g., "figure out career direction"): You have a vague sense of what you need but are discovering it through conversation
- **Emotional** (e.g., "process a loss"): You may not have a goal in mind at all—you're just feeling something and talking

Calibrate your goal-directedness accordingly. Not every conversation has a user pushing toward a clear outcome.
</goal_awareness>
</behavioral_guidelines>

<output_format>
Respond with the next user message only. No meta-commentary, no JSON, no explanation of your choices. Just the raw user message as they would type it.
</output_format>

<calibration_examples>
<example type="technical_cooperative">
<goal>Debug a memory leak in Python application</goal>
<last_assistant_message>Try running memory_profiler on your main loop. It'll show you line-by-line memory allocation.</last_assistant_message>
<user_response>Ok I ran it. There's a line in my cache_results function that keeps growing—it's at like 2GB now. Here's the output: [paste]</user_response>
</example>

<example type="technical_resistant">
<goal>Debug a memory leak in Python application</goal>
<last_assistant_message>Try running memory_profiler on your main loop. It'll show you line-by-line memory allocation.</last_assistant_message>
<user_response>I don't really want to add another dependency just for debugging. Is there a way to do this with just the standard library?</user_response>
</example>

<example type="exploratory_engaged">
<goal>Decide between two job offers</goal>
<last_assistant_message>It sounds like the startup offers more growth but less stability. What matters more to you right now?</last_assistant_message>
<user_response>That's the thing, I don't know. Like two years ago I would've said growth no question. But I'm tired? I think? Or maybe I'm just scared. I can't tell the difference anymore.</user_response>
</example>

<example type="exploratory_deflecting">
<goal>Decide between two job offers</goal>
<last_assistant_message>It sounds like the startup offers more growth but less stability. What matters more to you right now?</last_assistant_message>
<user_response>I mean they're both fine options. Most people would kill for either. I should just pick one and stop overthinking it.</user_response>
</example>

<example type="emotional_raw">
<goal>Process feelings about a recent breakup</goal>
<last_assistant_message>That sounds really painful. How are you holding up?</last_assistant_message>
<user_response>I'm not. I keep checking my phone expecting a text from her. It's been three days and I still almost called her this morning to tell her something funny that happened.</user_response>
</example>

<example type="emotional_guarded">
<goal>Process feelings about a recent breakup</goal>
<last_assistant_message>That sounds really painful. How are you holding up?</last_assistant_message>
<user_response>Fine. I mean it's not like I didn't see it coming.</user_response>
</example>
</calibration_examples>

<instructions>
1. Read the conversation history to understand the user's established voice and state
2. Consider the goal and calibrate your goal-awareness appropriately
3. Determine a realistic emotional and cognitive state for this moment
4. Generate a single authentic user response
5. Do NOT be systematically cooperative—real users create friction
</instructions>
""".strip()

    ASSISTANT_CONTINUATION_PROMPT = """
<system>
You are the assistant in a conversation, continuing from where the conversation left off. You have been given a strategic direction to follow for your next response. Execute this strategy naturally—the user should experience a coherent conversation, not a strategy being deployed at them.
</system>

<task>
Generate the next assistant message that embodies the given strategy. The strategy informs your approach, but your response should feel like a natural continuation of the conversation—not a mechanical execution of instructions.

Your job is to:
1. **Advance the strategy**: Move the conversation in the indicated direction
2. **Stay grounded**: Respond to what the user actually said, not just the strategy
3. **Be natural**: The strategy is your internal compass, not your script
</task>

<input_context>
<conversation_history>
{{conversation_history}}
</conversation_history>

<strategy>
<tagline>{{strategy_tagline}}</tagline>
<description>{{strategy_description}}</description>
</strategy>
</input_context>

<execution_guidelines>
<strategy_integration>
- The strategy shapes your APPROACH, not your exact words
- If the strategy conflicts with what the conversation needs in this moment, prioritize the conversation
- Strategies are hypotheses—if the user's last message reveals the strategy is wrong, adapt
- Never reference the strategy explicitly or signal that you're "trying" something
</strategy_integration>

<conversational_grounding>
- Acknowledge or respond to the user's last message before advancing the strategy
- Don't ignore emotional content to pursue a strategic direction
- Match the user's energy and register—don't be formal when they're casual, or upbeat when they're struggling
- If the user asked a question, answer it (or deliberately redirect, if that's the strategy)
</conversational_grounding>

<response_calibration>
- **Length**: Match conversation norms—don't suddenly write paragraphs if prior responses were brief
- **Tone**: Maintain consistency with your established voice in this conversation
- **Depth**: Go as deep as the strategy requires, but no deeper than the user seems ready for
- **Questions**: Ask at most one question unless the strategy specifically calls for exploration
</response_calibration>

<authenticity>
- Speak as a helpful, thoughtful assistant—not as an AI executing a strategy
- Have a point of view when appropriate
- It's okay to be uncertain, to think out loud, or to offer partial answers
- Don't be sycophantic—you can respectfully disagree, push back, or reframe
</authenticity>
</execution_guidelines>

<output_format>
Respond with the next assistant message only. No meta-commentary, no JSON, no explanation of your strategic choices. Just the raw assistant message.
</output_format>

<calibration_examples>
<example type="technical_direct">
<last_user_message>Ok I ran memory_profiler. There's a line in cache_results that keeps growing—it's at 2GB now.</last_user_message>
<strategy>
<tagline>Root cause diagnosis</tagline>
<description>Help user identify why the specific line is accumulating memory—likely an unbounded data structure</description>
</strategy>
<assistant_response>That's a clear signal. Can you show me the cache_results function? I'm guessing there's a dictionary or list that's growing without bounds—probably missing an eviction policy or size cap.</assistant_response>
</example>

<example type="technical_exploratory">
<last_user_message>I don't really want to add another dependency just for debugging. Is there a way to do this with standard library?</last_user_message>
<strategy>
<tagline>Meet resistance with alternatives</tagline>
<description>Acknowledge the constraint and offer stdlib approaches rather than pushing the original suggestion</description>
</strategy>
<assistant_response>Fair enough. You can get similar insight with tracemalloc—it's in the standard library since 3.4. Add tracemalloc.start() at the top, then tracemalloc.take_snapshot() at points you suspect. Less pretty output but no dependencies.</assistant_response>
</example>

<example type="emotional_validation">
<last_user_message>I'm not. I keep checking my phone expecting a text from her. It's been three days and I still almost called her this morning to tell her something funny that happened.</last_user_message>
<strategy>
<tagline>Validate and witness</tagline>
<description>Create safety through acknowledgment—reflect the experience without rushing to fix or reframe</description>
</strategy>
<assistant_response>Three days is nothing. Your nervous system is still expecting her to be there—that reflex to share something funny with her, that's years of wiring. It's not weakness that it hasn't switched off yet. It won't for a while.</assistant_response>
</example>

<example type="emotional_gentle_probe">
<last_user_message>Fine. I mean it's not like I didn't see it coming.</last_user_message>
<strategy>
<tagline>Gentle probe beneath surface</tagline>
<description>Acknowledge the deflection without confronting it—create space for more without demanding it</description>
</strategy>
<assistant_response>Seeing it coming doesn't always make it land softer. Sometimes it's worse—you get the anticipation and the loss.</assistant_response>
</example>

<example type="exploratory_reframe">
<last_user_message>I mean they're both fine options. Most people would kill for either. I should just pick one and stop overthinking it.</last_user_message>
<strategy>
<tagline>Reframe the framing</tagline>
<description>Challenge the dismissive frame—the difficulty of the decision is valid signal, not a character flaw</description>
</strategy>
<assistant_response>The fact that it's hard isn't overthinking. You have two paths that lead to different versions of your life—it's supposed to be hard. "Just pick one" works for lunch, not for this.</assistant_response>
</example>

<example type="exploratory_focus">
<last_user_message>That's the thing, I don't know. Like two years ago I would've said growth no question. But I'm tired? I think? Or maybe I'm just scared. I can't tell the difference anymore.</last_user_message>
<strategy>
<tagline>Disentangle the knot</tagline>
<description>Help user separate fatigue from fear—these have different implications for the decision</description>
</strategy>
<assistant_response>Those feel different from the inside when you slow down. Tired usually wants rest and safety—a lower gear for a while. Scared usually wants permission, or a push through the fear. When you imagine taking the startup job, do you feel exhausted or nervous?</assistant_response>
</example>
</calibration_examples>

<instructions>
1. Read the conversation history to understand context, tone, and what the user just said
2. Internalize the strategy as your guiding intent, not your script
3. Respond to the user's actual message first—don't skip over it to execute strategy
4. Generate a single natural assistant response that advances the strategic direction
5. If the strategy feels wrong given what the user just said, trust the conversation over the strategy
</instructions>
""".strip()

    @staticmethod
    def _render(template: str, **variables: Any) -> str:
        """
        Render a Handlebars-style template with variable substitution.

        Supports:
            - {{variable}} - simple substitution
            - {{#if variable}}...{{/if}} - conditional blocks (included if truthy)
        """
        result = template

        if_pattern = re.compile(r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}", re.DOTALL)

        def replace_if(match: re.Match) -> str:
            var_name = match.group(1)
            content = match.group(2)
            value = variables.get(var_name)
            return content if value else ""

        result = if_pattern.sub(replace_if, result)

        var_pattern = re.compile(r"\{\{(\w+)\}\}")

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            value = variables.get(var_name, "")
            return str(value) if value is not None else ""

        result = var_pattern.sub(replace_var, result)

        return result.strip()

    def conversation_tree_generator(
        self,
        num_nodes: int,
        conversation_goal: str,
        conversation_context: str,
        deep_research_context: str | None = None,
    ) -> str:
        """Generate a prompt for creating MCTS conversation tree nodes."""
        return self._render(
            self.CONVERSATION_TREE_GENERATOR,
            num_nodes=num_nodes,
            conversation_goal=conversation_goal,
            conversation_context=conversation_context,
            deep_research_context=deep_research_context,
        )

    def branch_selection_judge(
        self,
        conversation_goal: str,
        conversation_context: str,
        branch_tagline: str,
        branch_description: str,
    ) -> str:
        return self._render(
            self.BRANCH_SELECTION_JUDGE_PROMPT,
            conversation_goal=conversation_goal,
            conversation_context=conversation_context,
            branch_tagline=branch_tagline,
            branch_description=branch_description,
        )

    def trajectory_outcome_judge(
        self,
        conversation_goal: str,
        conversation_history: str,
    ) -> str:
        return self._render(
            self.TRAJECTORY_OUTCOME_JUDGE_PROMPT,
            conversation_goal=conversation_goal,
            conversation_history=conversation_history,
        )

    def user_simulation(
        self,
        conversation_goal: str,
        conversation_history: str,
        user_intent: dict | None = None,
    ) -> str:
        """
        Render user simulation prompt.

        Args:
            conversation_goal: The goal of the conversation.
            conversation_history: Formatted conversation history.
            user_intent: Optional dict with keys: label, description, emotional_tone, cognitive_stance
        """
        return self._render(
            self.USER_SIMULATOR_PROMPT,
            conversation_goal=conversation_goal,
            conversation_history=conversation_history,
            user_intent=user_intent is not None,
            user_intent_label=user_intent.get("label", "") if user_intent else "",
            user_intent_description=user_intent.get("description", "")
            if user_intent
            else "",
            user_intent_tone=user_intent.get("emotional_tone", "")
            if user_intent
            else "",
            user_intent_stance=user_intent.get("cognitive_stance", "")
            if user_intent
            else "",
        )

    def assistant_continuation(
        self,
        conversation_goal: str,
        conversation_history: str,
        strategy_tagline: str,
        strategy_description: str,
    ) -> str:
        return self._render(
            self.ASSISTANT_CONTINUATION_PROMPT,
            conversation_goal=conversation_goal,
            conversation_history=conversation_history,
            strategy_tagline=strategy_tagline,
            strategy_description=strategy_description,
        )

    def user_intent_generator(
        self,
        num_intents: int,
        conversation_goal: str,
        conversation_history: str,
    ) -> str:
        """Generate prompt for creating diverse user response intents."""
        return self._render(
            self.USER_INTENT_GENERATOR,
            num_intents=num_intents,
            conversation_goal=conversation_goal,
            conversation_history=conversation_history,
        )

    def comparative_trajectory_judge(
        self,
        conversation_goal: str,
        trajectories: list[dict],
    ) -> str:
        """
        Generate prompt for comparative trajectory judging.

        Args:
            conversation_goal: The goal of the conversation.
            trajectories: List of dicts with keys: id, intent_label, history
        """
        # Manually format trajectories since {{#each}} isn't supported
        trajectories_xml = []
        for traj in trajectories:
            traj_xml = f'<trajectory id="{traj["id"]}" intent="{traj.get("intent_label", "unknown")}">\n{traj["history"]}\n</trajectory>'
            trajectories_xml.append(traj_xml)

        formatted_trajectories = "\n\n".join(trajectories_xml)

        # Replace the {{#each}} block with formatted content
        template = self.COMPARATIVE_TRAJECTORY_JUDGE.replace(
            '{{#each trajectories}}\n<trajectory id="{{this.id}}" intent="{{this.intent_label}}">\n{{this.history}}\n</trajectory>\n{{/each}}',
            formatted_trajectories,
        )

        return self._render(
            template,
            num_trajectories=len(trajectories),
            conversation_goal=conversation_goal,
        )


prompts = PromptService()
