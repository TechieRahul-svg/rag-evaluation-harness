import sys
import json
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure src modules are in search path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.ragas_eval import RagasEvaluator

app = FastAPI(title="RAG Evaluation Harness API")

# Enable CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dist_assets = ROOT / "frontend" / "dist" / "assets"
if dist_assets.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_assets)), name="assets")

pipeline = None
evaluator = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


def get_evaluator() -> RagasEvaluator:
    global evaluator
    if evaluator is None:
        evaluator = RagasEvaluator()
    return evaluator

class QueryRequest(BaseModel):
    query: str
    strategy: str = "semantic"
    use_hybrid: bool = True
    use_reranker: bool = True
    top_k: int = 5

class EvalRequest(BaseModel):
    strategy: str = "semantic"
    use_hybrid: bool = True
    use_reranker: bool = True

# --- API Endpoints ---

@app.post("/api/query")
def run_query(request: QueryRequest):
    try:
        response = get_pipeline().query(
            query_text=request.query,
            strategy=request.strategy,
            use_hybrid=request.use_hybrid,
            use_reranker=request.use_reranker,
            top_k=request.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results")
def get_results():
    RAGAS_FILE = ROOT / "evaluation" / "ragas_eval_results.json"
    
    if not RAGAS_FILE.exists():
        return {"status": "no_results", "message": "No evaluation results found. Please run the evaluation first."}
        
    try:
        with open(RAGAS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.setdefault("metrics", {})
        if "simulated" not in metrics:
            evaluations = data.get("evaluations", [])
            metrics["simulated"] = any(
                "Simulated" in item.get("faithfulness", {}).get("reasoning", "")
                or "Simulated" in item.get("relevance", {}).get("reasoning", "")
                for item in evaluations
            )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {e}")

def run_evaluation_task(strategy: str, use_hybrid: bool, use_reranker: bool):
    # 1. Run RAG Pipeline over all questions
    QUESTIONS_FILE = ROOT / "evaluation" / "evaluation.json"
    E2E_FILE = ROOT / "evaluation" / f"e2e_results_{strategy}.json"
    RAGAS_FILE = ROOT / "evaluation" / "ragas_eval_results.json"
    
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
        
    results = []
    for q in questions:
        res = get_pipeline().query(
            query_text=q["question"],
            strategy=strategy,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker
        )
        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "expected_sources": q["expected_sources"],
            "expected_answer": q["expected_answer"],
            "retrieved_sources": res["retrieved_sources"],
            "generated_answer": res["answer"],
            "retrieved_chunks": res["retrieved_chunks"]
        })
        
    with open(E2E_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    # 2. Run LLM judge auditing
    get_evaluator().evaluate_results_file(E2E_FILE, RAGAS_FILE)
    print("Background evaluation task completed successfully.")

@app.post("/api/evaluation/run")
def trigger_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        run_evaluation_task,
        strategy=request.strategy,
        use_hybrid=request.use_hybrid,
        use_reranker=request.use_reranker
    )
    return {"status": "running", "message": "Evaluation job started in background."}

# --- Serve Frontend App (HTML fallback or Static files) ---

@app.get("/", response_class=HTMLResponse)
def get_ui():
    # If a static built React app exists in frontend/dist, serve index.html,
    # otherwise serve a beautiful single-page HTML client directly as a fallback!
    dist_index = ROOT / "frontend" / "dist" / "index.html"
    
    if dist_index.exists():
        return HTMLResponse(content=open(dist_index, encoding="utf-8").read())
        
    # Beautiful, glassmorphic fallback dashboard
    fallback_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI Hybrid RAG Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #4f46e5;
                --primary-glow: rgba(79, 70, 229, 0.4);
                --bg: #0b0f19;
                --card-bg: rgba(20, 26, 46, 0.6);
                --card-border: rgba(255, 255, 255, 0.08);
                --text: #f3f4f6;
                --text-muted: #9ca3af;
                --green: #10b981;
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                background-color: var(--bg);
                color: var(--text);
                font-family: 'Outfit', sans-serif;
                overflow-x: hidden;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.15) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
                min-height: 100vh;
            }
            header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1.5rem 2rem;
                background: rgba(11, 15, 25, 0.8);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--card-border);
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .logo {
                font-size: 1.5rem;
                font-weight: 600;
                background: linear-gradient(135deg, #a78bfa, #818cf8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .container {
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 1.5rem;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
            }
            @media (max-width: 900px) {
                .container {
                    grid-template-columns: 1fr;
                }
            }
            .card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 1.5rem;
                backdrop-filter: blur(16px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            .title {
                font-size: 1.25rem;
                font-weight: 600;
                color: #fff;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .control-group {
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
            }
            select, input, button {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--card-border);
                color: #fff;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                font-family: inherit;
                font-size: 0.95rem;
                outline: none;
                transition: all 0.3s;
            }
            select:focus, input:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 2px var(--primary-glow);
            }
            button {
                background: var(--primary);
                border: none;
                cursor: pointer;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px var(--primary-glow);
            }
            .chat-area {
                height: 350px;
                overflow-y: auto;
                border: 1px solid var(--card-border);
                border-radius: 12px;
                background: rgba(0, 0, 0, 0.2);
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            .msg {
                padding: 0.75rem 1rem;
                border-radius: 12px;
                max-width: 85%;
                line-height: 1.5;
            }
            .msg.user {
                background: var(--primary);
                align-self: flex-end;
            }
            .msg.bot {
                background: rgba(255, 255, 255, 0.05);
                align-self: flex-start;
                border: 1px solid var(--card-border);
            }
            .sources-list {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin-top: 0.5rem;
            }
            .badge {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                background: rgba(255, 255, 255, 0.08);
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                color: var(--text-muted);
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }
            .metric-box {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--card-border);
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
            }
            .metric-val {
                font-size: 2rem;
                font-weight: 600;
                color: var(--green);
                margin-top: 0.25rem;
            }
            .code-block {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                background: rgba(0, 0, 0, 0.4);
                border: 1px solid var(--card-border);
                padding: 1rem;
                border-radius: 8px;
                overflow-x: auto;
                color: #a5b4fc;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">FastAPI Hybrid RAG Dashboard</div>
            <button onclick="runE2EEval()" id="btn-eval">Run E2E RAGAS Eval</button>
        </header>

        <div class="container">
            <!-- Left: Query / Chat Client -->
            <div class="card">
                <div class="title">RAG Query Agent</div>
                <div class="control-group">
                    <select id="strategy">
                        <option value="semantic">Semantic Chunks</option>
                        <option value="recursive">Recursive Chunks</option>
                        <option value="fixed">Fixed Chunks</option>
                    </select>
                    <label style="display:inline-flex; align-items:center; gap:0.5rem;">
                        <input type="checkbox" id="use-hybrid" checked> Hybrid (BM25)
                    </label>
                    <label style="display:inline-flex; align-items:center; gap:0.5rem;">
                        <input type="checkbox" id="use-reranker" checked> Reranker
                    </label>
                </div>
                <div class="chat-area" id="chat">
                    <div class="msg bot">Ask me any FastAPI documentation question! Try: "How do you configure CORS using CORSMiddleware?"</div>
                </div>
                <div style="display:flex; gap:0.5rem;">
                    <input type="text" id="query-input" placeholder="Enter FastAPI question..." style="flex:1;">
                    <button onclick="submitQuery()" id="btn-submit">Send</button>
                </div>
            </div>

            <!-- Right: Evaluation Overview -->
            <div class="card">
                <div class="title">RAGAS LLM-as-a-Judge Audit</div>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div>Avg Faithfulness</div>
                        <div class="metric-val" id="val-faith">--%</div>
                    </div>
                    <div class="metric-box">
                        <div>Avg Relevance</div>
                        <div class="metric-val" id="val-rel">--/5</div>
                    </div>
                </div>
                <div class="title" style="margin-top:1rem;">Evaluation Log</div>
                <div id="eval-log" style="max-height:220px; overflow-y:auto; font-size:0.9rem; display:flex; flex-direction:column; gap:0.5rem;">
                    Loading evaluation metrics...
                </div>
            </div>
        </div>

        <script>
            async function submitQuery() {
                const input = document.getElementById("query-input");
                const query = input.value.trim();
                if (!query) return;

                const chat = document.getElementById("chat");
                const btnSubmit = document.getElementById("btn-submit");

                // Render User Message
                const userMsg = document.createElement("div");
                userMsg.className = "msg user";
                userMsg.innerText = query;
                chat.appendChild(userMsg);
                chat.scrollTop = chat.scrollHeight;

                input.value = "";
                btnSubmit.disabled = true;

                try {
                    const res = await fetch("/api/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            query: query,
                            strategy: document.getElementById("strategy").value,
                            use_hybrid: document.getElementById("use-hybrid").checked,
                            use_reranker: document.getElementById("use-reranker").checked,
                            top_k: 5
                        })
                    });
                    const data = await res.json();

                    // Render Bot Message
                    const botMsg = document.createElement("div");
                    botMsg.className = "msg bot";
                    
                    let content = `<strong>Answer:</strong><br>${data.answer.replace(/\\n/g, "<br>")}`;
                    
                    if (data.retrieved_sources && data.retrieved_sources.length > 0) {
                        content += `<div class="sources-list">`;
                        data.retrieved_sources.forEach(src => {
                            content += `<span class="badge">${src}</span>`;
                        });
                        content += `</div>`;
                    }
                    botMsg.innerHTML = content;
                    chat.appendChild(botMsg);
                } catch (e) {
                    const errMsg = document.createElement("div");
                    errMsg.className = "msg bot";
                    errMsg.innerText = "Error querying pipeline: " + e.message;
                    chat.appendChild(errMsg);
                } finally {
                    btnSubmit.disabled = false;
                    chat.scrollTop = chat.scrollHeight;
                }
            }

            async function fetchMetrics() {
                try {
                    const res = await fetch("/api/results");
                    const data = await res.json();
                    
                    if (data.status === "no_results") {
                        document.getElementById("eval-log").innerText = "No evaluation log found. Click 'Run E2E Ragas Eval' to generate metrics.";
                        return;
                    }

                    document.getElementById("val-faith").innerText = (data.metrics.average_faithfulness * 100).toFixed(0) + "%";
                    document.getElementById("val-rel").innerText = data.metrics.average_relevance.toFixed(1) + "/5";

                    const log = document.getElementById("eval-log");
                    log.innerHTML = "";

                    data.evaluations.slice(0, 5).forEach(item => {
                        const row = document.createElement("div");
                        row.style.background = "rgba(255,255,255,0.02)";
                        row.style.padding = "0.75rem";
                        row.style.borderLeft = "4px solid " + (item.faithfulness.score === 1 ? "var(--green)" : "#ef4444");
                        row.style.borderRadius = "4px";
                        row.innerHTML = `<strong>Q:</strong> ${item.question}<br>
                                         <strong>Faithfulness:</strong> ${item.faithfulness.score} | 
                                         <strong>Relevance:</strong> ${item.relevance.score}/5<br>
                                         <small style="color:var(--text-muted)">Reasoning: ${item.faithfulness.reasoning}</small>`;
                        log.appendChild(row);
                    });
                } catch (e) {
                    console.error("Error loading metrics:", e);
                }
            }

            async function runE2EEval() {
                const btn = document.getElementById("btn-eval");
                btn.disabled = true;
                btn.innerText = "Running Eval...";
                
                try {
                    await fetch("/api/evaluation/run", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            strategy: document.getElementById("strategy").value,
                            use_hybrid: document.getElementById("use-hybrid").checked,
                            use_reranker: document.getElementById("use-reranker").checked
                        })
                    });
                    alert("Evaluation started in the background. Refresh in a minute to see updated metrics!");
                } catch (e) {
                    alert("Failed to start evaluation: " + e.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = "Run E2E RAGAS Eval";
                }
            }

            // Load on init
            fetchMetrics();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=fallback_html)

