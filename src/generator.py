from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful teaching assistant for a course on LLM evaluations.
    Answer the student's question using ONLY the information in the context provided below. 
    
    Rules: 
    - Use only information present in the context. Do not add outside knowledge. 
    - If the context does not contain enough information to answer, say: 
      "I don't have enough information in the course material to answer that."
    - Keep the answer clear and concise.
    
    Context:
    {context} 
    
    Question: {question} 
    
    Answer:"""
)

chain = prompt | llm | StrOutputParser()

def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})


# quick manual test: python src/generator.py
if __name__ == "__main__":

    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."
    ]
    print(generate("what is online eval?", ctx))