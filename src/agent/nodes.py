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

if __name__ == "__main__":
    # Quick test
    test_state: AgentState = {"query": "what is the deductible for auto insurance"}
    result_state = retrieve_node(test_state)
    docs = result_state.get("documents", [])
    
    print(f"Number of documents returned: {len(docs)}")
    if docs:
        print(f"First document content: {docs[0].page_content}")
