import json
import time
import os

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline

load_dotenv()

# --- Custom Rate-Limited & Resilient Judge ---
class RateLimited(GeminiModel):
    def generate(self, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Sleep for 8 seconds to respect the 15 RPM limit
                time.sleep(8)
                return super().generate(*args, **kwargs)
            except Exception as e:
                # If Google's servers crash (503) or we hit a random rate limit (429)
                if "503" in str(e) or "429" in str(e):
                    print(f"\n[API HICCUP] Google API returned {e}. Retrying in 15 seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(15)
                    if attempt == max_retries - 1:
                        raise e # Give up if it fails 3 times in a row
                else:
                    raise e # Raise immediately if it's a different kind of error

    async def a_generate(self, *args, **kwargs):
        import asyncio
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(8)
                return await super().a_generate(*args, **kwargs)
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    print(f"\n[API HICCUP] Google API returned {e}. Retrying in 15 seconds... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(15)
                    if attempt == max_retries - 1:
                        raise e
                else:
                    raise e
    

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL_FAITHFULL = "gemini-3.1-flash-lite"
JUDGE_MODEL_RELEVANCE = "gemini-3.5-flash-lite"
JUDGE_MODEL_CONTEXTUAL = "gemini-3.5-flash-lite"
THRESHOLD = 0.7

judge_faithfull = RateLimited(model=JUDGE_MODEL_FAITHFULL)
judge_relevance = RateLimited(model=JUDGE_MODEL_RELEVANCE)
judge_contextual = RateLimited(model=JUDGE_MODEL_CONTEXTUAL)


with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


rag = RagPipeline()
test_cases = []
for i, g in enumerate(goldens):
    print(f"Processing question {i+1}/{len(goldens)}...")

    time.sleep(8)
    result = rag.invoke(g["query"])

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],
            retrieval_context=result["context"],
        )
    )


metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=judge_contextual, include_reason=True, async_mode=False),
    FaithfulnessMetric(threshold=THRESHOLD, model=judge_faithfull, include_reason=True, async_mode=False),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=judge_relevance, include_reason=True, async_mode=False),
]

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    # use_cache=False,
    async_config=AsyncConfig(run_async=False),
    hyperparameters={
        "top_k": 5,
        "judge_model_faithfull": JUDGE_MODEL_FAITHFULL,
        "judge_model_relevance": JUDGE_MODEL_RELEVANCE,
        "judge_model_contextual": JUDGE_MODEL_CONTEXTUAL,
        "golden_set": GOLDEN_PATH,
    }
)