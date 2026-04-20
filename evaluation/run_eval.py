import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict

from core.config import Settings
from core.graph_builder import build_financial_graph
from core.runner import create_llm, run_financial_query
from evaluation.scorer import Scorer

def load_benchmark(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def run_eval(benchmark_path: str, mode: str = "agent", output_dir: str = "evaluation/results", resume: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    settings = Settings()
    llm = create_llm(settings)
    compiled = build_financial_graph(llm)
    scorer = Scorer(settings)
    
    benchmark = load_benchmark(benchmark_path)
    
    # Handling Resume
    existing_ids = set()
    results = []
    # Use a fixed date-hour timestamp so resume works within the same hour
    timestamp = datetime.now().strftime("%Y%m%d_%H") 
    output_file = os.path.join(output_dir, f"eval_{mode}_{timestamp}.jsonl")
    
    if resume and os.path.exists(output_file):
        print(f"Resuming from existing file: {output_file}")
        with open(output_file, "r") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                existing_ids.add(item["id"])
                results.append(item)
    
    print(f"--- Starting {mode.upper()} Evaluation ({len(benchmark)} items) ---")
    
    scores = {"all": [], "numeric": [], "risk": [], "news_sentiment": [], "earnings_sentiment": []}
    
    # Initialize scores with existing results
    for r in results:
        scores["all"].append(r["score"])
        scores[r["type"]].append(r["score"])

    for i, item in enumerate(benchmark):
        if item["id"] in existing_ids:
            print(f"[{i+1}/{len(benchmark)}] Skipping {item['id']} (already processed)")
            continue
            
        print(f"[{i+1}/{len(benchmark)}] Querying: {item['question']}")
        
        # Prevent Rate Limit (30 RPM)
        time.sleep(3) 
        
        start_time = time.time()
        if mode == "agent":
            agent_result = run_financial_query(compiled, item['question'])
            prediction = agent_result.get("memo") or (agent_result["steps"][-1]["content"] if agent_result["steps"] else "No output")
            tool_calls = [s['node'] for s in agent_result['steps']]
        else:
            # Baseline Mode: Vanilla LLM call
            res = llm.invoke(item['question'])
            prediction = res.content
            tool_calls = ["None (Baseline)"]
            
        latency = time.time() - start_time
        
        score = 0.0
        reason = ""
        
        if item['type'] == "numeric":
            score = scorer.score_numeric(prediction, item['expected_answer'])
            reason = "Exact numeric match (1% tolerance)" if score == 1.0 else "Numeric mismatch"
        else:
            score, reason = scorer.score_qualitative(item['question'], prediction, item['expected_answer'])
            
        result_item = {
            "id": item['id'],
            "type": item['type'],
            "question": item['question'],
            "expected": item['expected_answer'],
            "prediction": prediction,
            "score": score,
            "reason": reason,
            "latency": round(latency, 2),
            "tool_calls": tool_calls,
            "mode": mode
        }
        
        results.append(result_item)
        scores["all"].append(score)
        scores[item['type']].append(score)
        
        # Write to file immediately
        with open(output_file, "a") as f:
            f.write(json.dumps(result_item) + "\n")
            
    # Aggregates
    summary = {
        "timestamp": timestamp,
        "mode": mode,
        "total_items": len(benchmark),
        "overall_accuracy": sum(scores["all"]) / len(benchmark) if benchmark else 0,
        "per_type_accuracy": {k: (sum(v)/len(v) if v else 0) for k, v in scores.items() if k != "all"},
        "avg_latency": sum([r['latency'] for r in results]) / len(results) if results else 0
    }
    
    summary_file = os.path.join(output_dir, f"summary_{mode}_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n--- {mode.upper()} Evaluation Complete ---")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="evaluation/benchmark_v1.jsonl")
    parser.add_argument("--mode", type=str, choices=["agent", "baseline"], default="agent", help="Evaluation mode")
    parser.add_argument("--resume", action="store_true", help="Resume from existing hourly result file")
    args = parser.parse_args()
    
    # Check if benchmark exists, if not use candidates for testing
    path = args.benchmark
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Defaulting to candidates.jsonl for testing.")
        path = "evaluation/candidates.jsonl"
        
    run_eval(path, mode=args.mode, resume=args.resume)
