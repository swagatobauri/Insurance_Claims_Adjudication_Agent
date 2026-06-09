"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

export default function Home() {
  const [claimText, setClaimText] = useState("");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [hasEvaluated, setHasEvaluated] = useState(false);
  
  const [decision, setDecision] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [retryCount, setRetryCount] = useState(0);
  const [webFallback, setWebFallback] = useState(false);
  
  const [loadingPhase, setLoadingPhase] = useState(0);
  const loadingPhrases = [
    "Retrieving policy documents...",
    "Cross-referencing state regulations...",
    "Grading semantic relevance...",
    "Synthesizing decision...",
    "Finalizing grounded response..."
  ];

  const handleEvaluate = async () => {
    if (!claimText.trim() || isEvaluating) return;
    
    setIsEvaluating(true);
    setHasEvaluated(false);
    setDecision("");
    setReasoning("");
    setSources([]);
    setLoadingPhase(0);
    
    // Start cycler
    const cycler = setInterval(() => {
      setLoadingPhase((prev) => (prev + 1) % loadingPhrases.length);
    }, 4000);
    
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
              if (parsed.type === "metadata") {
                setDecision(parsed.data.decision);
                setSources(parsed.data.sources);
                setRetryCount(parsed.data.retry_count);
                setWebFallback(parsed.data.web_fallback_used);
                isReasoningStream = true; // Everything after this is actual reasoning words
                setHasEvaluated(true);
              } else if (parsed.type === "token") {
                setHasEvaluated(true);
                if (isReasoningStream) {
                  setReasoning((prev) => prev + parsed.data);
                } else if (parsed.data.includes("Error:")) {
                  // Catch backend crashes and rate limits so the UI doesn't hang!
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

  const getDecisionColor = (d: string) => {
    const dec = d.toLowerCase();
    if (dec === "approve") return "text-approve";
    if (dec === "deny") return "text-deny";
    if (dec === "escalate") return "text-escalate";
    return "text-primary";
  };

  return (
    <main className="min-h-screen flex flex-col items-center selection:bg-accent selection:text-white">
      {/* SECTION 1 - HERO */}
      <section className="w-full max-w-4xl min-h-screen flex flex-col justify-center px-6 md:px-12 relative">
        <div className="mb-2">
          <span className="text-secondary text-[10px] uppercase tracking-[0.2em]">
            AI-Powered · Policy Grounded · Auditable
          </span>
        </div>
        
        <h1 className="text-6xl md:text-8xl font-light tracking-tight text-accent leading-[1.1] mb-12">
          Claims Adjudication <br /> Agent
        </h1>
        
        <div className="w-full h-px bg-border mb-16" />
        
        {/* SECTION 2 - INPUT */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12 w-full">
          <div className="md:col-span-8 flex flex-col">
            <textarea
              className="w-full bg-transparent border-b border-border focus:border-accent outline-none resize-none pb-4 text-xl md:text-2xl font-light transition-colors placeholder:text-secondary/50"
              rows={4}
              placeholder="Describe the claim details..."
              value={claimText}
              onChange={(e) => setClaimText(e.target.value)}
            />
            <button
              onClick={handleEvaluate}
              disabled={isEvaluating || !claimText.trim()}
              className="mt-6 bg-accent hover:bg-[#2A5278] text-white py-4 px-8 uppercase text-sm tracking-widest font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isEvaluating ? "Evaluating..." : "Evaluate Claim"}
            </button>
          </div>
          
          <div className="md:col-span-4 flex items-start">
            <p className="text-secondary text-sm leading-relaxed">
              Our deterministic LangGraph agent evaluates claims securely against the policy database. 
              It provides a rigorous, grounded decision backed directly by source citations without hallucination.
            </p>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 text-secondary/50 flex flex-col items-center gap-2 animate-pulse">
          <span className="text-[10px] uppercase tracking-widest">Results</span>
          <ChevronDown size={16} strokeWidth={1} />
        </div>
      </section>

      {/* SECTION 3 - RESULTS */}
      {hasEvaluated && (
        <section className="w-full max-w-4xl px-6 md:px-12 py-24 animate-fade-in">
          <div className="flex items-center gap-6 mb-16">
            <div className="flex-1 h-px bg-border" />
            <span className="text-secondary text-xs uppercase tracking-widest">Assessment</span>
            <div className="flex-1 h-px bg-border" />
          </div>

          <div className="flex flex-col">
            {/* ROW 1 */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-8 py-8 border-b border-border items-baseline">
              <div className="md:col-span-3">
                <span className="text-secondary text-xs uppercase tracking-widest">Decision</span>
              </div>
              <div className="md:col-span-9">
                <h2 className={`text-4xl md:text-5xl font-light uppercase tracking-wide ${getDecisionColor(decision)}`}>
                  {decision || "..."}
                </h2>
              </div>
            </div>

            {/* ROW 2 */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-8 py-12 border-b border-border">
              <div className="md:col-span-3 pt-2">
                <span className="text-secondary text-xs uppercase tracking-widest">Reasoning</span>
              </div>
              <div className="md:col-span-9">
                {isEvaluating && !decision ? (
                  <div className="flex flex-col gap-4 py-4 animate-pulse">
                    <div className="flex items-center gap-4">
                      <div className="w-4 h-4 rounded-full border-2 border-accent border-t-transparent animate-spin" />
                      <p className="text-xl text-accent font-medium tracking-wide">
                        {loadingPhrases[loadingPhase]}
                      </p>
                    </div>
                    <div className="h-2 w-full max-w-md bg-border rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-accent transition-all duration-1000 ease-in-out"
                        style={{ width: `${((loadingPhase + 1) / loadingPhrases.length) * 100}%` }}
                      />
                    </div>
                  </div>
                ) : (
                  <p className="text-lg md:text-xl font-light leading-[1.8] whitespace-pre-wrap break-words">
                    {reasoning}
                  </p>
                )}
              </div>
            </div>

            {/* ROW 3 */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-8 py-12">
              <div className="md:col-span-3 pt-2">
                <span className="text-secondary text-xs uppercase tracking-widest">Sources</span>
              </div>
              <div className="md:col-span-9">
                {sources.length > 0 ? (
                  <div className="flex flex-col gap-3 mb-10">
                    {sources.map((src, idx) => (
                      <div key={idx} className="font-mono text-[11px] md:text-xs text-secondary bg-border/30 px-3 py-2 border border-border/50 truncate">
                        {src}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-secondary text-sm italic mb-10">No sources retrieved.</p>
                )}
                
                <div className="flex flex-wrap gap-4">
                  <div className="border border-border rounded-full px-4 py-1.5 text-xs text-secondary">
                    Retrieval Attempts: <span className="text-primary font-medium">{retryCount}</span>
                  </div>
                  <div className="border border-border rounded-full px-4 py-1.5 text-xs text-secondary">
                    Web Fallback: <span className="text-primary font-medium">{webFallback ? "Yes" : "No"}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* SECTION 4 - FOOTER */}
      <footer className="w-full border-t border-border mt-auto">
        <div className="max-w-4xl mx-auto px-6 py-8 text-center">
          <p className="text-secondary text-xs tracking-wide">
            AI-generated assessment. Must be reviewed by a licensed adjuster before any action is taken.
          </p>
        </div>
      </footer>
    </main>
  );
}
