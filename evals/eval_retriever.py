import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.retriever import build_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL_RECALL = "gemini-3.1-flash-lite"
JUDGE_MODEL_PRECISION = "gemini-3.5-flash-lite"
THRESHOLD = 0.7

judge_recall = GeminiModel(model=JUDGE_MODEL_RECALL)
judge_precision = GeminiModel(model=JUDGE_MODEL_PRECISION)

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


retriever = build_retriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)"
        )
    )


metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=judge_recall, include_reason=True, async_mode=False),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=judge_precision, include_reason=True, async_mode=False),
]


evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
    hyperparameters={
        "retriever": "base_k5",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model_recall": JUDGE_MODEL_RECALL,
        "judge_model_precision": JUDGE_MODEL_PRECISION,
        "golden_set": GOLDEN_PATH,
    }
)