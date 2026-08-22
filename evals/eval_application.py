import json
import time
import os

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline


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
    

GOLDEN_PATH = "goldens/correctness_goldens.json"
JUDGE_MODEL_CORRECTNESS = "gemini-3.1-flash-lite"
JUDGE_MODEL_COMPLETENESS = "gemini-3.5-flash-lite"
JUDGE_MODEL_STYLE = "gemini-3.5-flash-lite"
THRESHOLD = 0.7

judge_correctness = RateLimited(model=JUDGE_MODEL_CORRECTNESS)
judge_completeness = RateLimited(model=JUDGE_MODEL_COMPLETENESS)
judge_style = RateLimited(model=JUDGE_MODEL_STYLE)


with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


rag = RagPipeline()
test_cases = []
for i, g in enumerate(goldens):
    print(f"Processing question {i+1}/{len(goldens)}...")

    time.sleep(8)
    result = rag.invoke(g["question"])

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],
            expected_output=g["ideal_answer"],
        )
    )


correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
        "Do NOT deduct for brevity, missing elaboration, or omitted points — omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
        Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
        Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=judge_correctness,
    strict_mode=False,
    async_mode=False,
)

# 3b. COMPLETENESS — reference-based, judges COVERAGE (not correctness)
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly — factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
        Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=judge_completeness,
    strict_mode=False,
    async_mode=False
)

# 3c. STYLE — reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
        "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
        "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
        "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
        "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
        "Do NOT reward or penalize based on correctness, completeness, or length — only on style and tone.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes."),
        Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
        Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
        Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=THRESHOLD,
    model=judge_style,
    strict_mode=False,
    async_mode=False,
)


evaluate(test_cases=test_cases, metrics=[correctness, completeness, style], async_config=AsyncConfig(run_async=False))