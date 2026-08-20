from src.reranker import RerankingRetriever
from src.generator import generate

class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    def invoke(self, query: str) -> dict:
        docs = self.retriever.invoke(query)

        context = [doc.page_content for doc in docs]

        answer = generate(query, context)

        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("what is drift and why does it matter after deployment?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")