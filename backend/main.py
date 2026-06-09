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
# pyrefly: ignore [missing-import]
from fastapi import UploadFile, File, HTTPException
import shutil
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import FastEmbedEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS

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
        
        # Poll with a hard 90-second global timeout
        MAX_WAIT_SECONDS = 90
        elapsed = 0
        while elapsed < MAX_WAIT_SECONDS:
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                break
            except asyncio.TimeoutError:
                elapsed += 1
                # Send elapsed time as a heartbeat so the frontend can show a live timer
                yield f"data: {json.dumps({'type': 'heartbeat', 'elapsed': elapsed})}\n\n"
        else:
            # Timed out — kill the process and return an error
            proc.kill()
            await proc.wait()
            yield f"data: {json.dumps({'type': 'timeout', 'data': 'The pipeline exceeded the 90-second limit. Groq API may be rate-limited. Please wait 60 seconds and try again.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
                
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
            "web_fallback_used": final_state.get("web_fallback_used", False),
            "relevance_score": final_state.get("relevance_score", ""),
            "hallucination_score": final_state.get("hallucination_score", ""),
            "source_documents": final_state.get("source_documents", [])
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

@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "temp"))
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        persist_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "faiss_index"))
        
        if os.path.exists(persist_directory):
            vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(chunks)
        else:
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            
        vectorstore.save_local(persist_directory)
        
        return {"status": "success", "message": f"Successfully ingested {len(chunks)} chunks from {file.filename}."}
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
