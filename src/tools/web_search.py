import os
import sys
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Ensure the root of the project is in the Python path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# pyrefly: ignore [missing-import]
from langchain_community.tools.tavily_search import TavilySearchResults
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from src.agent.state import AgentState

# Load environment variables (including TAVILY_API_KEY)
load_dotenv()

def web_search_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    documents = state.get("documents", [])
    
    if not query:
        print("WEB FALLBACK: No query provided")
        return {"web_fallback_used": True}
        
    # Initialize Tavily search tool
    tool = TavilySearchResults(max_results=3)
    
    # Run the search
    try:
        results = tool.invoke({"query": query})
        
        web_docs = []
        for result in results:
            content = result.get("content", "")
            url = result.get("url", "unknown")
            doc = Document(
                page_content=content,
                metadata={"source": url, "type": "web_fallback"}
            )
            web_docs.append(doc)
            
        print(f"WEB FALLBACK: found {len(web_docs)} results for query: {query}")
        
        # Append web documents to existing documents
        updated_documents = documents + web_docs
        
        return {
            "documents": updated_documents,
            "web_fallback_used": True
        }
    except Exception as e:
        print(f"WEB FALLBACK: Error during search - {e}")
        return {"web_fallback_used": True}

if __name__ == "__main__":
    # Quick test
    print("--- RUNNING WEB SEARCH NODE ---")
    test_query = "California state regulations homeowners insurance claim denial"
    initial_state: AgentState = {
        "query": test_query,
        "documents": [],
        "web_fallback_used": False
    }
    
    result_state = web_search_node(initial_state)
    docs = result_state.get("documents", [])
    
    print(f"Documents added: {len(docs)}")
    if docs:
        print(f"First result source URL: {docs[0].metadata.get('source')}")
