import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [strategy, setStrategy] = useState("semantic");
  const [useHybrid, setUseHybrid] = useState(true);
  const [useReranker, setUseReranker] = useState(true);
  const [chatHistory, setChatHistory] = useState([
    {
      role: "bot",
      text: "Ask me any FastAPI documentation question! Try: 'How do you declare HTTP header parameters?'",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [evalResults, setEvalResults] = useState(null);
  const [evalRunning, setEvalRunning] = useState(false);

  // Fetch LLM-as-a-judge evaluation results
  const fetchEvalResults = async () => {
    try {
      const res = await fetch("/api/results");
      const data = await res.json();
      if (data.status !== "no_results") {
        setEvalResults(data);
      }
    } catch (e) {
      console.error("Failed to fetch evaluation results", e);
    }
  };

  useEffect(() => {
    fetchEvalResults();
  }, []);

  const handleSend = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    const userMessage = { role: "user", text: query };
    setChatHistory((prev) => [...prev, userMessage]);
    setQuery("");

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          strategy,
          use_hybrid: useHybrid,
          use_reranker: useReranker,
          top_k: 5,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Query request failed");
      }
      
      const botMessage = {
        role: "bot",
        text: data.answer,
        sources: data.retrieved_sources,
        chunks: data.retrieved_chunks,
      };
      setChatHistory((prev) => [...prev, botMessage]);
    } catch (e) {
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: "Error querying the pipeline: " + e.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const startEvaluation = async () => {
    setEvalRunning(true);
    try {
      const res = await fetch("/api/evaluation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy,
          use_hybrid: useHybrid,
          use_reranker: useReranker,
        }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Evaluation request failed");
      }
      alert("Evaluation triggered in background. Please refresh in a minute!");
    } catch (e) {
      alert("Failed to start evaluation: " + e.message);
    } finally {
      setEvalRunning(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <span className="logo-glow"></span>
          <h1>FastAPI Hybrid RAG Dashboard</h1>
        </div>
        <button 
          onClick={startEvaluation} 
          disabled={evalRunning} 
          className="btn btn-secondary"
        >
          {evalRunning ? "Auditing..." : "Run RAGAS Judge Audit"}
        </button>
      </header>

      {/* Main Grid */}
      <div className="main-grid">
        {/* Sidebar Controls */}
        <aside className="sidebar">
          <div className="card">
            <h2>RAG Configurations</h2>
            
            <div className="control-group">
              <label>Retrieval Strategy</label>
              <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                <option value="semantic">Semantic Chunks</option>
                <option value="recursive">Recursive Chunks</option>
                <option value="fixed">Fixed Chunks</option>
              </select>
            </div>

            <div className="toggle-group">
              <label className="toggle">
                <input 
                  type="checkbox" 
                  checked={useHybrid} 
                  onChange={(e) => setUseHybrid(e.target.checked)} 
                />
                <span className="toggle-label">Enable Hybrid Search (BM25)</span>
              </label>
            </div>

            <div className="toggle-group">
              <label className="toggle">
                <input 
                  type="checkbox" 
                  checked={useReranker} 
                  onChange={(e) => setUseReranker(e.target.checked)} 
                />
                <span className="toggle-label">Enable Re-ranking (Cross-Encoder)</span>
              </label>
            </div>
          </div>

          {/* Context Inspector */}
          <div className="card context-card">
            <h2>Retrieved Chunks Inspector</h2>
            {selectedChunk ? (
              <div className="inspector-content">
                <div className="meta-row">
                  <span className="badge">{selectedChunk.source}</span>
                </div>
                <div className="score-row">
                  <span>RRF: <strong>{selectedChunk.rrf_score.toFixed(5)}</strong></span>
                  <span>Rerank: <strong>{selectedChunk.rerank_score.toFixed(4)}</strong></span>
                </div>
                <pre className="chunk-text">{selectedChunk.text}</pre>
              </div>
            ) : (
              <p className="placeholder-text">Ask a question and click on a retrieved source badge below to inspect its source context.</p>
            )}
          </div>
        </aside>

        {/* Chat & Ragas Metrics Column */}
        <main className="content-area">
          {/* Chat Panel */}
          <div className="card chat-card">
            <div className="chat-messages">
              {chatHistory.map((msg, idx) => (
                <div key={idx} className={`message-bubble ${msg.role}`}>
                  <p>{msg.text}</p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-list">
                      {msg.chunks.map((chunk, cIdx) => (
                        <span 
                          key={cIdx} 
                          onClick={() => setSelectedChunk(chunk)} 
                          className="source-badge"
                          title="Click to inspect context"
                        >
                          {chunk.source.split("/").pop()}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="message-bubble bot loading">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              )}
            </div>
            
            <div className="chat-input-bar">
              <input 
                type="text" 
                value={query} 
                onChange={(e) => setQuery(e.target.value)} 
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder="Ask FastAPI questions..." 
                disabled={loading}
              />
              <button onClick={handleSend} disabled={loading} className="btn btn-primary">
                Send
              </button>
            </div>
          </div>

          {/* Judge Metrics */}
          <div className="card metrics-card">
            <h2>RAGAS LLM-as-a-Judge Audit Summary</h2>
            {evalResults?.metrics?.simulated && (
              <p className="placeholder-text">These metrics are simulated because GROQ_API_KEY was not available when the audit ran.</p>
            )}
            {evalResults ? (
              <div className="metrics-layout">
                <div className="metrics-summary">
                  <div className="metric-box">
                    <span>Avg Faithfulness</span>
                    <div className="metric-value">
                      {(evalResults.metrics.average_faithfulness * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="metric-box">
                    <span>Avg Answer Relevance</span>
                    <div className="metric-value">
                      {evalResults.metrics.average_relevance.toFixed(1)}/5
                    </div>
                  </div>
                </div>

                <div className="audit-log">
                  <h3>Evaluation Feed (Latest Audited Queries)</h3>
                  <div className="log-scroll">
                    {evalResults.evaluations.slice(0, 5).map((item, idx) => (
                      <div key={idx} className={`log-row ${item.faithfulness.score === 1 ? "pass" : "fail"}`}>
                        <div className="log-header">
                          <strong>Q: {item.question}</strong>
                          <span className="log-badges">
                            F: {item.faithfulness.score} | R: {item.relevance.score}/5
                          </span>
                        </div>
                        <p className="log-reason">Reasoning: {item.faithfulness.reasoning}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <p className="placeholder-text">No evaluation results available yet. Run the RAGAS Judge Audit to view automated RAG benchmarks.</p>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;

