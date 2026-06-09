import os
import sys
import concurrent.futures
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Ensure the root of the project is in the Python path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import Chroma
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.retrievers import BM25Retriever
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Literal, List

from src.agent.state import AgentState

# Load environment variables
load_dotenv()

# Initialize global vector store and retrievers
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
persist_directory = os.path.join(base_dir, ".chroma")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Extract documents from Chroma to initialize BM25
db_data = vectorstore.get()
db_docs = db_data.get('documents', [])
db_metadatas = db_data.get('metadatas', [])

all_documents = []
for doc, meta in zip(db_docs, db_metadatas):
    all_documents.append(Document(page_content=doc, metadata=meta or {}))

bm25_retriever = None
if all_documents:
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 4

def retrieve_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    
    if not query:
        print("RETRIEVE: No query provided in state.")
        return {"documents": []}
        
    dense_results = []
    keyword_results = []
    
    def run_dense():
        return vectorstore.similarity_search(query, k=4)
        
    def run_keyword():
        if bm25_retriever:
            return bm25_retriever.invoke(query)
        return []

    # Run two searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dense = executor.submit(run_dense)
        future_keyword = executor.submit(run_keyword)
        
        dense_results = future_dense.result()
        keyword_results = future_keyword.result()
        
    # Merge and deduplicate by page_content
    merged_docs = []
    seen_contents = set()
    
    for doc in dense_results + keyword_results:
        if doc.page_content not in seen_contents:
            merged_docs.append(doc)
            seen_contents.add(doc.page_content)
            
    print(f"RETRIEVE: Found {len(merged_docs)} chunks after hybrid merge")
    
    return {"documents": merged_docs}

def rewrite_query_node(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count", 0)
    
    if retry_count >= 2:
        return state
        
    claim = state.get("claim", "")
    query = state.get("query", "")
    
    # Using the active Groq model
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    prompt = f"""You are an expert insurance policy researcher. Given this insurance claim and a weak search query that failed to retrieve relevant policy documents, rewrite the query to be more specific, use policy terminology, and maximize the chance of finding relevant coverage clauses.

Claim: {claim}
Current query: {query}

Return only the rewritten query, nothing else."""

    result = llm.invoke(prompt)
    new_query = result.content.strip()
    
    print(f"QUERY REWRITER: retry {retry_count} — rewritten query: {new_query}")
    
    return {"query": new_query, "retry_count": retry_count + 1}

class AgentDecision(BaseModel):
    decision: Literal["approve", "deny", "escalate"] = Field(description="Decision on the claim")
    reasoning: str = Field(description="Detailed explanation grounded in the documents")
    sources: List[str] = Field(description="List of source document names used")

def decision_node(state: AgentState) -> AgentState:
    relevance_score = state.get("relevance_score", "")
    retry_count = state.get("retry_count", 0)
    claim = state.get("claim", "")
    documents = state.get("documents", [])
    
    if relevance_score == "not_relevant" and retry_count >= 2:
        print("DECISION NODE: decision = escalate")
        return {
            "decision": "escalate",
            "reasoning": "Insufficient policy evidence found after multiple retrieval attempts. Routing to human adjuster.",
            "sources": []
        }
        
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    structured_llm = llm.with_structured_output(AgentDecision)
    
    docs_text = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in documents])
    
    prompt = f"""You are an expert insurance claims adjudicator. 

Here is the insurance claim submitted:
{claim}

Here are the relevant policy documents retrieved:
{docs_text}

Based strictly on the policy documents above:
1. Decide whether to approve, deny, or escalate this claim
2. Provide clear reasoning citing specific policy clauses
3. List the exact source document names you used

Return a JSON with these exact fields:
- decision: one of 'approve', 'deny', 'escalate'
- reasoning: detailed explanation grounded in the documents
- sources: list of source document names used"""

    messages = [{"role": "user", "content": prompt}]
    
    try:
        result = structured_llm.invoke(messages)
        print(f"DECISION NODE: decision = {result.decision}")
        return {
            "decision": result.decision,
            "reasoning": result.reasoning,
            "sources": result.sources
        }
    except Exception as e:
        print(f"Error making decision: {e}")
        return {
            "decision": "escalate",
            "reasoning": f"Failed to generate decision: {str(e)}",
            "sources": []
        }

if __name__ == "__main__":
    # Quick test
    test_state: AgentState = {"query": "what is the deductible for auto insurance"}
    result_state = retrieve_node(test_state)
    docs = result_state.get("documents", [])
    
    print(f"Number of documents returned: {len(docs)}")
    if docs:
        print(f"First document content: {docs[0].page_content}")
        
    # Test query rewriter
    print("\n--- RUNNING QUERY REWRITER ---")
    rewrite_state: AgentState = {
        "claim": "My roof was damaged during a storm",
        "query": "roof damage",
        "retry_count": 0
    }
    rewrite_result = rewrite_query_node(rewrite_state)
    print(f"New query: {rewrite_result.get('query')}")
    print(f"Retry count: {rewrite_result.get('retry_count')}")

    # Test decision node
    print("\n--- RUNNING DECISION NODE ---")
    decision_query = "windshield damage falling object coverage"
    decision_initial_state: AgentState = {"query": decision_query, "claim": decision_query}
    decision_retrieved_state = retrieve_node(decision_initial_state)
    
    decision_state: AgentState = {
        "claim": "My car windshield was cracked by a falling tree branch, claiming replacement cost",
        "query": decision_query,
        "relevance_score": "relevant",
        "retry_count": 0,
        "documents": decision_retrieved_state.get("documents", [])
    }
    
    final_decision_state = decision_node(decision_state)
    print(f"Decision: {final_decision_state.get('decision')}")
    print(f"Reasoning: {final_decision_state.get('reasoning')}")
    print(f"Sources: {final_decision_state.get('sources')}")
