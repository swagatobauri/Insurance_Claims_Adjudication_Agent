import os
import sys
from typing import Literal
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

# Ensure the root of the project is in the Python path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agent.state import AgentState

# Load environment variables (including GROQ_API_KEY)
load_dotenv()

# Define the Pydantic model for structured output
class GradeRelevance(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    score: Literal["yes", "no"] = Field(
        description="Does this document contain information relevant to adjudicating this claim? Answer yes or no."
    )

def relevance_grader_node(state: AgentState) -> AgentState:
    claim = state.get("claim", "")
    documents = state.get("documents", [])
    
    if not claim or not documents:
        print("RELEVANCE GRADER: 0/0 documents relevant — score: not_relevant")
        return {"relevance_score": "not_relevant"}
        
    # Initialize the LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeRelevance)
    
    system_prompt = """You are a grader assessing relevance of a retrieved document to an insurance claim.
If the document contains keywords, conditions, coverage details, exclusions, or rules related to the user's claim, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out completely irrelevant documents."""

    yes_count = 0
    total_docs = len(documents)
    
    for doc in documents:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Claim: {claim}\n\nDocument: {doc.page_content}\n\nDoes this document contain information relevant to adjudicating this claim? Answer yes or no."}
        ]
        
        try:
            result = structured_llm_grader.invoke(messages)
            if result.score == "yes":
                yes_count += 1
        except Exception as e:
            print(f"Error grading document: {e}")
            pass
            
    # Set relevance score based on at least 2 "yes" grades
    final_score = "relevant" if yes_count >= 2 else "not_relevant"
    
    print(f"RELEVANCE GRADER: {yes_count}/{total_docs} documents relevant — score: {final_score}")
    
    return {"relevance_score": final_score}

if __name__ == "__main__":
    from src.agent.nodes import retrieve_node
    
    # Quick test
    test_claim = "My car was damaged in a hailstorm, I am claiming repair costs"
    
    # Run retrieval first
    # Using claim text as query for the retrieval step
    initial_state: AgentState = {"query": test_claim, "claim": test_claim}
    retrieved_state = retrieve_node(initial_state)
    
    # Build state for grader
    state_for_grader: AgentState = {
        "claim": test_claim,
        "query": test_claim,
        "documents": retrieved_state.get("documents", [])
    }
    
    # Run relevance grader
    print("--- RUNNING RELEVANCE GRADER ---")
    result_state = relevance_grader_node(state_for_grader)
    
    print(f"Output relevance score: {result_state.get('relevance_score')}")
