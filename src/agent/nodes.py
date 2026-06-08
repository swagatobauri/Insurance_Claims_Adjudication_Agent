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
