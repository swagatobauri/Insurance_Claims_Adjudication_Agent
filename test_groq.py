import sys
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing Groq API...", flush=True)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=0)
try:
    res = llm.invoke("Say hello")
    print("Success:", res.content, flush=True)
except Exception as e:
    print("Error:", e, flush=True)
