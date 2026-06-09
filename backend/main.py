import sys
import os
import json
import asyncio
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    claim: str

@app.post("/evaluate")
async def evaluate_claim(request: ClaimRequest):
    async def event_generator():
        # IMMEDIATELY yield a status update so the frontend resolves the fetch promise and shows activity.
        # This prevents the UI from appearing "stuck" while LangGraph and PyTorch are initializing.
        yield f"data: {json.dumps({'type': 'token', 'data': 'Initializing isolated agent environment (PyTorch/FAISS)'})}\n\n"
        
        env = os.environ.copy()
        # Strictly enforce OpenMP settings in the subprocess to prevent Apple Silicon deadlocks
        env["OMP_NUM_THREADS"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["TOKENIZERS_PARALLELISM"] = "false"
        
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "agent", "graph.py"))
        
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path, "--claim", request.claim, "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )
        
        # While the agent is running, stream dots to the frontend
        # We must not block on wait() while stdout fills up, but stdout is tiny (<1KB) so it won't overflow
        while True:
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'token', 'data': '.'})}\n\n"
                
        stdout_data = await proc.stdout.read()
        
        if proc.returncode != 0:
            yield f"data: {json.dumps({'type': 'token', 'data': '\n\nError: Agent failed to execute. Check terminal logs.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
            
        stdout_str = stdout_data.decode()
        try:
            json_str = stdout_str.split("###JSON_START###")[1].split("###JSON_END###")[0].strip()
            final_state = json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse subprocess output: {stdout_str}")
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        
        # Clear the initializing string
        yield f"data: {json.dumps({'type': 'token', 'data': '\n\n'})}\n\n"
        
        # Send metadata
        metadata = {
            "decision": final_state.get("decision", ""),
            "sources": final_state.get("sources", []),
            "retry_count": final_state.get("retry_count", 0),
            "web_fallback_used": final_state.get("web_fallback_used", False)
        }
        yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
        
        # Stream reasoning
        reasoning = final_state.get("reasoning", "")
        tokens = reasoning.split(" ")
        for i, token in enumerate(tokens):
            word = token if i == 0 else " " + token
            yield f"data: {json.dumps({'type': 'token', 'data': word})}\n\n"
            await asyncio.sleep(0.04)
            
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
