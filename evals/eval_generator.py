import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL_FAITHFULL = "gemini-3.1-flash-lite"
JUDGE_MODEL_RELEVANCE = "gemini-3.5-flash-lite"
THRESHOLD = 0.7

judge_faithfull = GeminiModel(model=JUDGE_MODEL_FAITHFULL)
judge_relevance = GeminiModel(model=JUDGE_MODEL_RELEVANCE)

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


test_cases = []
for g in goldens:
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


evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
    hyperparameters={
        "top_k": 5,
        "judge_model_faithfull": JUDGE_MODEL_FAITHFULL,
        "judge_model_relevance": JUDGE_MODEL_RELEVANCE,
        "golden_set": GOLDEN_PATH,
    }
)