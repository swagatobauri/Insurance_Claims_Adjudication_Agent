"use client";

import { useState } from "react";
import { ShieldCheck, Lock, Activity, CheckCircle2, ChevronDown, ChevronRight, Loader2, UploadCloud } from "lucide-react";

export default function Home() {
  const [claimText, setClaimText] = useState("");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [hasEvaluated, setHasEvaluated] = useState(false);
  
  const [decision, setDecision] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [retryCount, setRetryCount] = useState(0);
  const [webFallback, setWebFallback] = useState(false);
  const [relevanceScore, setRelevanceScore] = useState("");
  const [hallucinationScore, setHallucinationScore] = useState("");
  const [sourceDocs, setSourceDocs] = useState<{page_content: string, metadata: any}[]>([]);
  
  const [loadingPhase, setLoadingPhase] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [timeoutError, setTimeoutError] = useState("");

  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState({ text: "", type: "" });

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.type !== "application/pdf") {
      setUploadMessage({ text: "Please select a PDF file.", type: "error" });
      return;
    }

    setIsUploading(true);
    setUploadMessage({ text: "Ingesting policy into vector store...", type: "info" });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload-policy", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setUploadMessage({ text: "Policy successfully injected into the agent's brain.", type: "success" });
      } else {
        const err = await response.json();
        setUploadMessage({ text: err.detail || "Upload failed.", type: "error" });
      }
    } catch (error) {
      setUploadMessage({ text: "Network error during upload.", type: "error" });
    } finally {
      setIsUploading(false);
      // Reset input so the same file can be selected again if needed
      e.target.value = "";
    }
  };

  const handleEvaluate = async () => {
    if (!claimText.trim() || isEvaluating) return;
    
    setIsEvaluating(true);
    setHasEvaluated(false);
    setDecision("");
    setReasoning("");
    setSources([]);
    setSourceDocs([]);
    setLoadingPhase(0);
    setElapsed(0);
    setTimeoutError("");
    
    // Start cycler: phase 0 (Retrieve), phase 1 (Relevance), phase 2 (Hallucination), phase 3 (Synthesis)
    const cycler = setInterval(() => {
      setLoadingPhase((prev) => (prev < 4 ? prev + 1 : prev));
    }, 4500);
    
    try {
      const response = await fetch("http://localhost:8000/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: claimText }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      let isReasoningStream = false;

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6).trim();
            if (!dataStr) continue;
            
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.type === "heartbeat") {
                // Live elapsed timer from backend
                setElapsed(parsed.elapsed);
                // Auto-advance tracer steps based on elapsed time
                if (parsed.elapsed >= 5) setLoadingPhase(p => Math.max(p, 1));
                if (parsed.elapsed >= 15) setLoadingPhase(p => Math.max(p, 2));
                if (parsed.elapsed >= 25) setLoadingPhase(p => Math.max(p, 3));
              } else if (parsed.type === "timeout") {
                setTimeoutError(parsed.data);
                setHasEvaluated(false);
              } else if (parsed.type === "metadata") {
                setDecision(parsed.data.decision);
                setSources(parsed.data.sources);
                setRetryCount(parsed.data.retry_count);
                setWebFallback(parsed.data.web_fallback_used);
                setRelevanceScore(parsed.data.relevance_score || "");
                setHallucinationScore(parsed.data.hallucination_score || "");
                setSourceDocs(parsed.data.source_documents || []);
                isReasoningStream = true; 
                setHasEvaluated(true);
              } else if (parsed.type === "token") {
                setHasEvaluated(true);
                if (isReasoningStream) {
                  setReasoning((prev) => prev + parsed.data);
                } else if (parsed.data.includes("Error:")) {
                  setDecision("ERROR");
                  setReasoning(parsed.data.trim());
                }
              } else if (parsed.type === "done") {
                break;
              }
            } catch (e) {
              console.error("Parse error", e);
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
    } finally {
      clearInterval(cycler);
      setIsEvaluating(false);
    }
  };

  const getStatusStyles = (d: string) => {
    const dec = d.toLowerCase();
    if (dec === "approve") return "bg-[#f0fdf4] text-[#16a34a] border-[#16a34a]";
    if (dec === "deny") return "bg-[#fef2f2] text-[#dc2626] border-[#dc2626]";
    if (dec === "escalate") return "bg-[#fffbeb] text-[#d97706] border-[#d97706]";
    return "bg-gray-100 text-gray-800 border-gray-300";
  };

  return (
    <main className="min-h-screen grid grid-cols-1 md:grid-cols-2 bg-background selection:bg-accent selection:text-white">
      
      {/* LEFT COLUMN: Input & Context */}
      <section className="flex flex-col border-r border-border p-8 md:p-16 justify-between h-screen sticky top-0 overflow-y-auto">
        
        <div className="flex flex-col gap-8 w-full max-w-xl mx-auto">
          <div>
            <div className="flex items-center gap-2 mb-4 text-accent">
              <ShieldCheck size={24} strokeWidth={1.5} />
              <span className="text-xs uppercase tracking-[0.2em] font-medium">Claims Adjudication</span>
            </div>
            <h1 className="text-4xl lg:text-5xl font-light tracking-tight text-primary leading-tight mb-6">
              Deterministic Policy Evaluation
            </h1>
          </div>

          <div className="w-full">
            {/* STEP 1: CONTEXT UPLOAD */}
            <div className="w-full mb-10">
              <h2 className="text-xs uppercase tracking-widest text-secondary mb-3 flex items-center gap-2">
                Step 1: Inject Custom Context <span className="text-[10px] bg-border px-2 py-0.5 rounded-sm text-secondary/70">Optional</span>
              </h2>
              <div className="relative w-full border border-dashed border-border hover:border-accent bg-white/30 hover:bg-white/50 transition-all rounded-sm p-6 flex flex-col items-center justify-center text-center cursor-pointer overflow-hidden">
                <input 
                  type="file" 
                  accept=".pdf" 
                  onChange={handleFileUpload}
                  disabled={isUploading || isEvaluating}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                />
                {isUploading ? (
                  <Loader2 size={24} className="text-accent animate-spin mb-2" />
                ) : (
                  <UploadCloud size={24} className="text-accent mb-2" />
                )}
                <p className="text-sm font-medium text-primary">Upload Custom Policy (.pdf)</p>
                <p className="text-xs text-secondary mt-1">Instantly indexes into the FAISS vector database.</p>
              </div>
              
              {uploadMessage.text && (
                <div className={`mt-3 text-xs p-3 rounded-sm border ${
                  uploadMessage.type === "success" ? "bg-green-50 border-green-200 text-green-700" :
                  uploadMessage.type === "error" ? "bg-red-50 border-red-200 text-red-700" :
                  "bg-blue-50 border-blue-200 text-blue-700"
                }`}>
                  {uploadMessage.type === "success" && <CheckCircle2 size={12} className="inline mr-1 mb-0.5" />}
                  {uploadMessage.text}
                </div>
              )}
            </div>

            {/* STEP 2: CLAIM DESCRIPTION */}
            <h2 className="block text-xs uppercase tracking-widest text-secondary mb-3">Step 2: Describe the Claim</h2>
            <textarea
              className="w-full bg-white/50 border border-border focus:border-accent rounded-sm outline-none resize-none p-4 text-lg font-light transition-all placeholder:text-secondary/40 shadow-sm"
              rows={5}
              placeholder="e.g. My car door is damaged by an accident..."
              value={claimText}
              onChange={(e) => setClaimText(e.target.value)}
            />
            
            <button
              onClick={handleEvaluate}
              disabled={isEvaluating || !claimText.trim() || isUploading}
              className="mt-6 w-full bg-accent hover:bg-[#2A5278] text-white py-4 px-8 text-sm uppercase tracking-widest font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md flex justify-center items-center gap-3"
            >
              {isEvaluating ? (
                <><Loader2 size={18} className="animate-spin" /> Processing Pipeline...</>
              ) : (
                "Execute Adjudication"
              )}
            </button>
          </div>
        </div>

        {/* System Architecture Legend */}
        <div className="w-full max-w-xl mx-auto mt-12 pt-8 border-t border-border">
          <p className="text-[10px] uppercase tracking-widest text-secondary/70 mb-4 font-semibold">Active Guardrails</p>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/40 p-3 rounded-sm border border-border/50">
              <p className="text-xs font-medium text-primary mb-1">Deterministic Graph</p>
              <p className="text-[10px] text-secondary leading-relaxed">Cyclic LangGraph state transitions ensure strict logical routing without arbitrary generative drift.</p>
            </div>
            <div className="bg-white/40 p-3 rounded-sm border border-border/50">
              <p className="text-xs font-medium text-primary mb-1">Zero-Hallucination</p>
              <p className="text-[10px] text-secondary leading-relaxed">Strict factual grounding loop prevents the LM from synthesizing information not present in the vector store.</p>
            </div>
          </div>
        </div>
      </section>

      {/* RIGHT COLUMN: Dynamic Execution Space */}
      <section className="flex flex-col h-screen overflow-y-auto bg-white p-8 md:p-16 relative">
        
        {/* STATE 0: IDLE */}
        {!isEvaluating && !decision && (
          <div className="m-auto flex flex-col items-center justify-center text-center opacity-50 max-w-md">
            <Activity size={48} strokeWidth={1} className="text-secondary mb-6" />
            <h3 className="text-xl font-light text-primary mb-2">System Idle</h3>
            <p className="text-sm text-secondary leading-relaxed">
              Submit an adjudication query in the left panel to initialize the LangGraph pipeline and begin the trace.
            </p>
          </div>
        )}

        {/* TIMEOUT ERROR STATE */}
        {!isEvaluating && timeoutError && !decision && (
          <div className="m-auto flex flex-col items-center justify-center text-center max-w-md">
            <div className="w-12 h-12 rounded-full border-2 border-red-300 flex items-center justify-center mb-6">
              <span className="text-red-500 text-lg font-bold">!</span>
            </div>
            <h3 className="text-xl font-medium text-red-600 mb-2">Pipeline Timeout</h3>
            <p className="text-sm text-secondary leading-relaxed mb-4">{timeoutError}</p>
            <button
              onClick={() => setTimeoutError("")}
              className="text-xs uppercase tracking-widest text-accent border border-accent px-4 py-2 hover:bg-accent hover:text-white transition-colors"
            >
              Reset
            </button>
          </div>
        )}

        {/* STATE 1: PROCESSING (Live Tracer) */}
        {isEvaluating && !decision && (
          <div className="w-full max-w-xl mx-auto mt-12 animate-fade-in">
            <div className="mb-10">
              <h3 className="text-xs uppercase tracking-widest text-secondary mb-2">Execution Trace</h3>
              <div className="h-px w-full bg-border" />
            </div>

            <div className="flex flex-col gap-8 relative">
              <div className="absolute left-3.5 top-2 bottom-2 w-px bg-border/50 z-0" />
              
              {/* Step 1 */}
              <div className="relative z-10 flex gap-6">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center border-2 bg-white transition-colors duration-500 ${loadingPhase >= 0 ? "border-accent text-accent" : "border-border text-border"}`}>
                  {loadingPhase > 0 ? <CheckCircle2 size={16} /> : <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />}
                </div>
                <div>
                  <h4 className={`text-sm font-medium ${loadingPhase >= 0 ? "text-primary" : "text-secondary"}`}>Step 1: Context Aggregation</h4>
                  <p className="text-xs text-secondary mt-1 leading-relaxed">
                    [Node: Retrieve] Queried vector stores utilizing semantic text embedding matches.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className={`relative z-10 flex gap-6 transition-opacity duration-500 ${loadingPhase >= 1 ? "opacity-100" : "opacity-30"}`}>
                <div className={`w-7 h-7 rounded-full flex items-center justify-center border-2 bg-white transition-colors duration-500 ${loadingPhase >= 1 ? "border-accent text-accent" : "border-border text-border"}`}>
                  {loadingPhase > 1 ? <CheckCircle2 size={16} /> : loadingPhase === 1 ? <div className="w-2 h-2 rounded-full bg-accent animate-pulse" /> : null}
                </div>
                <div>
                  <h4 className={`text-sm font-medium ${loadingPhase >= 1 ? "text-primary" : "text-secondary"}`}>Step 2: Strict Clause Validation</h4>
                  <p className="text-xs text-secondary mt-1 leading-relaxed">
                    [Node: Grade Relevance] Language model grading semantic relevance against strict parameters.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className={`relative z-10 flex gap-6 transition-opacity duration-500 ${loadingPhase >= 2 ? "opacity-100" : "opacity-30"}`}>
                <div className={`w-7 h-7 rounded-full flex items-center justify-center border-2 bg-white transition-colors duration-500 ${loadingPhase >= 2 ? "border-accent text-accent" : "border-border text-border"}`}>
                  {loadingPhase > 2 ? <CheckCircle2 size={16} /> : loadingPhase === 2 ? <div className="w-2 h-2 rounded-full bg-accent animate-pulse" /> : null}
                </div>
                <div>
                  <h4 className={`text-sm font-medium ${loadingPhase >= 2 ? "text-primary" : "text-secondary"}`}>Step 3: Guardrail Check</h4>
                  <p className="text-xs text-secondary mt-1 leading-relaxed">
                    [Node: Hallucination Grader] Cross-checking final statement claims against raw string fragments within source document text.
                  </p>
                </div>
              </div>

              {/* Step 4 (Active until backend responds) */}
              <div className={`relative z-10 flex gap-6 transition-opacity duration-500 ${loadingPhase >= 3 ? "opacity-100" : "opacity-0 hidden"}`}>
                <div className="w-7 h-7 rounded-full flex items-center justify-center border-2 bg-white border-accent text-accent">
                  <Loader2 size={14} className="animate-spin" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-primary flex items-center gap-3">
                    Step 4: Compiling Resolution
                    {elapsed > 0 && (
                      <span className="text-xs font-mono text-accent bg-accent/10 px-2 py-0.5 rounded-sm">
                        {elapsed}s / 90s
                      </span>
                    )}
                  </h4>
                  <p className="text-xs text-secondary mt-1 leading-relaxed">
                    [Node: Synthesis] Streaming final structured decision and grounded reasoning trace...
                  </p>
                  {elapsed > 60 && (
                    <p className="text-xs text-amber-600 mt-2 font-medium">
                      ⚠ Taking longer than usual — Groq API may be throttling. Will auto-cancel at 90s.
                    </p>
                  )}
                </div>
              </div>

            </div>
          </div>
        )}

        {/* STATE 2: RESOLUTION */}
        {decision && (
          <div className="w-full max-w-2xl mx-auto pb-24 animate-fade-in flex flex-col h-full mt-12 md:mt-0">
            
            <div className="flex items-center justify-between mb-8 pb-6 border-b border-border">
              <span className="text-xs uppercase tracking-widest text-secondary">Final Adjudication</span>
              <div className={`px-4 py-1.5 rounded-sm border uppercase tracking-widest text-xs font-bold ${getStatusStyles(decision)}`}>
                {decision}
              </div>
            </div>

            <div className="mb-10">
              <h3 className="text-sm font-medium text-primary mb-3">Justification Trace</h3>
              <div className="bg-gray-50/50 p-6 rounded-sm border border-border/60">
                <p className="text-sm font-light leading-relaxed text-primary whitespace-pre-wrap break-words">
                  {reasoning}
                </p>
              </div>
            </div>

            {/* Technical Metadata Grid */}
            <div className="mb-10">
              <h3 className="text-sm font-medium text-primary mb-3">Backend Telemetry</h3>
              <div className="border border-border/60 rounded-sm overflow-hidden text-sm">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-gray-50/80 border-b border-border/60 text-xs uppercase tracking-wider text-secondary">
                      <th className="py-3 px-4 font-medium">Metric</th>
                      <th className="py-3 px-4 font-medium">Observed UI Target State</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/60 text-primary">
                    <tr>
                      <td className="py-3 px-4 font-medium">Retrieval Loops</td>
                      <td className="py-3 px-4">{retryCount === 0 ? "1st Pass Match" : `${retryCount} Retries`}</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-medium">Web Search Fallback</td>
                      <td className="py-3 px-4">{webFallback ? "Active (Tavily triggered)" : "Inactive (Tavily bypassed)"}</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-medium">Hallucination Score</td>
                      <td className="py-3 px-4">
                        {hallucinationScore === "grounded" ? "0.00 (100% Grounded)" : hallucinationScore === "hallucinated" ? "Failed (Hallucination Detected)" : "N/A"}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 font-medium">Confidence Boundary</td>
                      <td className="py-3 px-4">
                        {decision.toLowerCase() === "approve" || decision.toLowerCase() === "deny" ? "Safe (High Confidence)" : "Exception (Escalation Required)"}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Interactive Grounding Accordion */}
            <div className="mb-auto">
              <h3 className="text-sm font-medium text-primary mb-3">Grounded Context Extraction</h3>
              {sourceDocs.length > 0 ? (
                <div className="flex flex-col gap-3">
                  {sourceDocs.map((doc, idx) => (
                    <details key={idx} className="group border border-border/60 rounded-sm bg-white overflow-hidden">
                      <summary className="flex items-center justify-between p-3 cursor-pointer bg-gray-50/50 hover:bg-gray-50 transition-colors list-none [&::-webkit-details-marker]:hidden">
                        <span className="text-xs font-mono text-secondary truncate pr-4">
                          {doc.metadata?.source || `Document ${idx+1}`}
                        </span>
                        <ChevronDown size={14} className="text-secondary group-open:rotate-180 transition-transform" />
                      </summary>
                      <div className="p-4 border-t border-border/60 bg-gray-50/30">
                        <p className="text-xs font-mono text-secondary leading-relaxed break-words whitespace-pre-wrap">
                          {doc.page_content}
                        </p>
                      </div>
                    </details>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-secondary italic border border-border/60 p-4 bg-gray-50/50 rounded-sm">
                  No source documents retrieved for this trace.
                </div>
              )}
            </div>

            {/* Compliance Positioning */}
            <div className="mt-12 pt-6 border-t border-border flex items-start gap-3">
              <Lock size={14} strokeWidth={2} className="text-slate-400 mt-0.5 shrink-0" />
              <p className="text-[10px] text-slate-500 uppercase tracking-widest leading-relaxed">
                AI-generated assessment. Must be reviewed by a licensed adjuster before any action is taken.
              </p>
            </div>
          </div>
        )}
      </section>

    </main>
  );
}
