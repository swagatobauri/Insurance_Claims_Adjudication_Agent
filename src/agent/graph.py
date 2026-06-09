import os
import sys

# Ensure the root of the project is in the Python path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import retrieve_node, rewrite_query_node, decision_node, get_vectorstore
from src.agent.graders import relevance_grader_node, hallucination_grader_node
from src.tools.web_search import web_search_node

def route_after_relevance(state: AgentState) -> str:
    relevance_score = state.get("relevance_score", "")
    retry_count = state.get("retry_count", 0)
    
    if relevance_score == "relevant":
        return "decision"
    elif relevance_score == "not_relevant" and retry_count < 2:
        return "rewrite_query"
    else:
        return "web_search"

def route_after_hallucination(state: AgentState) -> str:
    hallucination_score = state.get("hallucination_score", "")
    
    if hallucination_score == "grounded":
        return END
    else:
        # If hallucinated, we try to generate the decision again
        return "decision"

# Initialize StateGraph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("relevance_grader", relevance_grader_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("decision", decision_node)
workflow.add_node("hallucination_grader", hallucination_grader_node)

# Add Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "relevance_grader")

# Add Conditional Edge for Relevance
workflow.add_conditional_edges(
    "relevance_grader",
    route_after_relevance,
    {
        "decision": "decision",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search"
    }
)

# Loop back
workflow.add_edge("rewrite_query", "retrieve")

workflow.add_edge("web_search", "decision")
workflow.add_edge("decision", "hallucination_grader")

# Add Conditional Edge for Hallucination
workflow.add_conditional_edges(
    "hallucination_grader",
    route_after_hallucination,
    {
        END: END,
        "decision": "decision"
    }
)

# Compile graph
graph = workflow.compile()

def run_claim(claim: str) -> AgentState:
    """Run the claim adjudication graph from start to finish."""
    
    # Pre-initialize singletons strictly in the main thread 
    # to avoid thread-safety bugs inside LangGraph's workers.
    get_vectorstore()
    
    initial_state: AgentState = {
        "claim": claim,
        "query": claim,
        "documents": [],
        "relevance_score": "",
        "hallucination_score": "",
        "decision": "",
        "reasoning": "",
        "sources": [],
        "retry_count": 0,
        "web_fallback_used": False
    }
    
    print(f"\n--- INITIATING CLAIM ADJUDICATION ---")
    print(f"Claim: {claim}\n")
    
    # Run graph
    final_state = graph.invoke(initial_state)
    
    return final_state

if __name__ == "__main__":
    # Test
    test_claim = "My car was damaged in a hailstorm last Tuesday. I am claiming full repair costs under my auto insurance policy."
    
    result = run_claim(test_claim)
    
    print("\n================ FINAL REPORT ================")
    print(f"Decision: {result.get('decision')}")
    print(f"Reasoning: {result.get('reasoning')}")
    print(f"Sources: {result.get('sources')}")
    print(f"Retry count: {result.get('retry_count')}")
    print(f"Web fallback used: {result.get('web_fallback_used')}")
    print("==============================================")
