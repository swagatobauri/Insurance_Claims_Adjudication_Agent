# pyrefly: ignore [missing-import]
import streamlit as st
import subprocess
import json
import sys
import os

st.set_page_config(
    page_title="Claims Adjudication Agent",
    page_icon="🛡️",
    layout="wide"
)

st.title("Insurance Claims Adjudication Agent")
st.caption("Deterministic · Policy Grounded · Zero-Hallucination · LangGraph")

st.divider()

with st.sidebar:
    st.header("📘 How It Works")
    st.markdown("""
    **Pipeline Steps:**
    1. `Retrieve` — FAISS semantic search
    2. `Grade Relevance` — LLM scores each chunk
    3. `Rewrite Query` — if retrieval is weak
    4. `Web Search` — Tavily fallback
    5. `Decision` — structured LLM output
    6. `Hallucination Check` — grounding loop
    """)
    st.divider()
    st.info("Powered by LangGraph + Groq llama-3.3-70b-versatile + FAISS")
    st.divider()
    st.header("📄 Upload Custom Policy")
    uploaded_file = st.file_uploader("Upload a PDF policy", type=["pdf"])
    if uploaded_file:
        import requests
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        try:
            r = requests.post("http://localhost:8000/upload-policy", files=files)
            if r.status_code == 200:
                st.success(f" {r.json().get('message', 'Uploaded successfully.')}")
            else:
                st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.warning(f"Backend not reachable: {e}. Make sure FastAPI is running on port 8000.")

claim = st.text_area(
    "📋 Describe the Insurance Claim",
    placeholder="e.g. My car door was damaged by a falling tree branch...",
    height=150
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    evaluate = st.button("Evaluate Claim", use_container_width=True, type="primary")

if evaluate:
    if not claim.strip():
        st.warning("Please enter a claim description.")
    else:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        steps = [
            "Step 1: Context Aggregation — querying FAISS vector store...",
            "Step 2: Grading Relevance — LLM scoring retrieved chunks...",
            "Step 3: Guardrail Check — cross-checking reasoning vs. source text...",
            "Step 4: Compiling Resolution — synthesizing final decision...",
        ]

        import time
        progress_bar = progress_placeholder.progress(0, text=steps[0])
        for i, step in enumerate(steps):
            progress_bar.progress((i + 1) * 25, text=step)
            time.sleep(0.5)

        with status_placeholder.status("Running full LangGraph pipeline...", expanded=True) as s:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["TOKENIZERS_PARALLELISM"] = "false"

            try:
                result = subprocess.run(
                    [sys.executable, "src/agent/graph.py", "--claim", claim, "--json"],
                    capture_output=True, text=True, timeout=120,
                    cwd=os.getcwd(), env=env
                )
                s.update(label="Pipeline complete!", state="complete")
            except subprocess.TimeoutExpired:
                s.update(label="Pipeline timed out after 120s", state="error")
                st.error("The pipeline exceeded the 120-second timeout. Groq may be rate-limiting. Please wait 60 seconds and try again.")
                st.stop()

        progress_placeholder.empty()
        status_placeholder.empty()

        if result.returncode != 0:
            st.error(f"Pipeline error. Check that your .env file has valid API keys.")
            with st.expander("Debug Output"):
                st.code(result.stderr[:2000])
        else:
            try:
                json_str = result.stdout.split("###JSON_START###")[1].split("###JSON_END###")[0].strip()
                data = json.loads(json_str)

                decision = data.get("decision", "").upper()
                reasoning = data.get("reasoning", "")
                sources = data.get("sources", [])
                retry_count = data.get("retry_count", 0)
                web_fallback = data.get("web_fallback_used", False)
                hallucination = data.get("hallucination_score", "")
                source_docs = data.get("source_documents", [])

                st.divider()

                # Decision badge
                if decision == "APPROVE":
                    st.success(f"## ✅ Decision: {decision}")
                elif decision == "DENY":
                    st.error(f"## ❌ Decision: {decision}")
                elif decision == "ESCALATE":
                    st.warning(f"## ⚠️ Decision: {decision} — Routed to Human Adjuster")
                else:
                    st.info(f"## Decision: {decision}")

                st.divider()

                # Telemetry
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval Loops", "1st Pass" if retry_count == 0 else f"{retry_count} Retries")
                col2.metric("Web Fallback", "✓ Active" if web_fallback else "✗ Bypassed")
                col3.metric("Hallucination Check", "✓ Grounded" if hallucination == "grounded" else "⚠ Detected")
                col4.metric("Source Docs Used", len(source_docs))

                st.divider()

                st.subheader("📝 Justification Trace")
                st.write(reasoning)

                if sources:
                    st.subheader("📌 Referenced Policies")
                    for src in sources:
                        st.markdown(f"- `{src}`")

                if source_docs:
                    st.subheader("📂 Grounded Source Extraction")
                    st.caption("Expand to see the exact policy clauses the agent used to make this decision.")
                    for i, doc in enumerate(source_docs):
                        label = doc.get("metadata", {}).get("source", f"Document {i+1}")
                        label = os.path.basename(label)
                        with st.expander(f"📄 {label}"):
                            st.code(doc.get("page_content", ""), language=None)

                st.divider()
                st.caption("⚖️ AI-generated assessment. Must be reviewed by a licensed adjuster before any action is taken.")

            except Exception as e:
                st.error(f"Failed to parse agent output: {e}")
                with st.expander("Raw Output"):
                    st.code(result.stdout[:2000])
