import asyncio
from backend.core.mcts import MCTSAgent
from backend.llm.client import LLM
from backend.utils.config import config

llm = LLM(
    api_key=config.llm_api_key.get_secret_value(),
    base_url=config.llm_base_url,
    model="z-ai/glm-4.7",
)


async def test_mcts():
    mcts_agent = MCTSAgent(
        llm=llm,
        goal="Figure out a first approach to use JePA for natural language processing",
        first_message="I'm trying to understand how world models like JePA can be applied to natural language processing. I'm not sure how to start. What does a dataset look like? How does it compare to the efficiency of transformers?",
        init_branch=6,
        deep_research=False,
        turns_per_branch=5,
    )

    result = await mcts_agent.run(rounds=2)
    result.save_json("mcts_output.json")


if __name__ == "__main__":
    asyncio.run(test_mcts())
