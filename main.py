import asyncio
import time

from backend.core.dts import DTSConfig, DTSEngine
from backend.llm.client import LLM
from backend.utils.config import config

llm = LLM(
    api_key=config.llm_api_key.get_secret_value(),
    base_url=config.llm_base_url,
    model="z-ai/glm-4.7",
)


async def test_dts():
    engine = DTSEngine(
        llm=llm,
        config=DTSConfig(
            goal="Identify a project idea that uses LLMs + multi-agent systems to create a fun and engaging game that cannot be done without agentic/LLMs. Avoid aggressive monetization (it'll be subscription based)",
            first_message="I want to design a monetizable game for the app-store that uses LLMs. Any ideas?",
            deep_research=False,
            turns_per_branch=2,
            user_intents_per_branch=2,
        ),
    )

    result = await engine.run(rounds=2)
    result.save_json("dts_output.json")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(test_dts())
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
