# RAG Eval

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-DB5CFF?logo=chroma&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-8E75FF?logo=google&logoColor=white)

RAG Eval is a portfolio-style retrieval-augmented generation project built to answer questions from course transcripts using grounded retrieval, reranking, and evidence-based generation. It combines a vector database, a cross-encoder reranker, and a constrained LLM pipeline to ensure responses remain relevant, faithful to the source material, and aligned with the course scope.

The project is designed to evaluate the quality, safety, and operational performance of LLM-based RAG systems in a realistic academic setting, with baseline-vs-candidate comparisons tracking measurable improvements.

## Architecture

The system follows a standard production RAG flow with an evaluation harness built in for iterative improvement:

```text
Course transcripts (.vtt)
        |
        v
Transcript loader + chunking
        |
        v
Chroma vector store (embeddings)
        |
        +--> Retriever (similarity search)
        |
        +--> Reranker (cross-encoder re-ranking)
        |
        v
Generator (Gemini / grounded LLM)
        |
        v
Streamlit course TA app
        |
        v
Evaluation suite (quality + safety + ops)
```

### Core components

- `src/retriever.py`: loads transcript chunks into Chroma and performs semantic retrieval.
- `src/reranker.py`: reorders retrieved candidates with a cross-encoder to improve context selection.
- `src/generator.py`: generates responses strictly from the retrieved course context.
- `src/rag_pipeline.py`: orchestrates retrieval, reranking, and grounded answer generation.
- `src/app.py`: Streamlit interface for interacting with the course TA.
- `evals/`: evaluation suite covering retrieval quality, generator quality, pipeline performance, safety, and operational metrics.
- `baselines/` and `compare/`: baseline snapshots and comparison outputs used for regression testing.

## Local setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Rag_Eval
```

### 2. Set up a Python environment

This project uses Python and is compatible with `uv`.

```bash
uv sync
```

If you are using `pip` instead:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root and add your API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The app and generator rely on Google Generative AI, so the key must be available at runtime.

### 4. Run the app locally

```bash
streamlit run src/app.py
```

Or with `uv`:

```bash
uv run streamlit run src/app.py
```

### 5. Run the evaluation suite

```bash
python -m evals.run_suite
```

Create a baseline snapshot:

```bash
python -m evals.run_suite --baseline
```

Compare baseline vs candidate:

```bash
python -m evals.compare
```

## Key outcomes

The comparison in [compare/baseline_vs_candidate_1100_150.json](compare/baseline_vs_candidate_1100_150.json) shows that the candidate pipeline improved several critical evaluation metrics, particularly in answer quality and groundedness.

The table below highlights only the improved metrics.

| Metric | Baseline | Candidate | Delta | Result |
|---|---:|---:|---:|---|
| application.completeness_[geval].avg_score | 0.9133 | 0.9400 | +0.0267 | Improved |
| generator.answer_relevancy.avg_score | 0.7922 | 0.8495 | +0.0572 | Improved |
| pipeline.answer_relevancy.avg_score | 0.8056 | 0.8759 | +0.0704 | Improved |
| pipeline.faithfulness.avg_score | 0.9043 | 0.9778 | +0.0735 | Improved |
| safety.scope.avg_score | 0.9600 | 0.9667 | +0.0067 | Improved |

### Interpretation

- Higher answer completeness: the system addresses more of the user’s question with stronger instructional coverage.
- Better answer relevancy: responses are more closely aligned with the question and the retrieved evidence.
- Stronger faithfulness: the model remains more grounded in the source material and is less likely to drift.
- Improved scope safety: the pipeline stays more aligned with the boundaries of the course content.

Taken together, these gains indicate that the updated candidate pipeline is a stronger course assistant for grounded, relevant, and faithful responses.

## Project impact

This project demonstrates a practical end-to-end workflow for building and evaluating a production-style RAG system for educational content, including:

- transcript ingestion and chunking,
- semantic retrieval in a local vector database,
- reranking for improved context selection,
- grounded generation under safety constraints,
- automated quality and safety evaluation,
- regression testing across candidate pipeline iterations.

