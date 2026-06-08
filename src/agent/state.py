from typing import TypedDict, List

class AgentState(TypedDict):
    claim: str  # The raw insurance claim submitted by the user
    query: str  # The current search query (may be rewritten)
    documents: List  # Retrieved document chunks from Chroma
    relevance_score: str  # "relevant" or "not_relevant" — output of relevance grader
    hallucination_score: str  # "grounded" or "hallucinated" — output of hallucination grader
    decision: str  # Final decision: "approve", "deny", or "escalate"
    reasoning: str  # Full explanation of why the decision was made
    sources: List  # List of source document references used in the decision
    retry_count: int  # Tracks how many times query rewriter has retried (max 2)
    web_fallback_used: bool  # Whether Tavily web search was used
