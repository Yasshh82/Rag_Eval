import json
import time
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

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
JUDGE_MODEL = "gemini-3.1-flash-lite"
JUDGE_MODEL_FAITHFULL = "gemini-3.1-flash-lite"
JUDGE_MODEL_RELEVANCE = "gemini-3.5-flash-lite"
THRESHOLD = 0.7

judge_model = RateLimited(model=JUDGE_MODEL)
judge_faithfull = RateLimited(model=JUDGE_MODEL_FAITHFULL)
judge_relevance = RateLimited(model=JUDGE_MODEL_RELEVANCE)

def run():
    # with open(GOLDEN_PATH) as f:
    #     goldens = json.load(f)
    goldens = load_goldens(GOLDEN_PATH)


    test_cases = []
    for i, g in enumerate(goldens):
        print(f"Processing question {i+1}/{len(goldens)}...")
        time.sleep(5)

        context = g["ideal_context"]
        answer = generate(g["query"], context)

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=context
            )
        )


    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge_faithfull, include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge_relevance, include_reason=True, async_mode=False),
    ]


    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())