"""
Evaluation API endpoints for RAGAS evaluation dashboard.
"""
import os
import json
import subprocess
import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/eval", tags=["evaluation"])

# Global state for evaluation status
eval_status = "idle"
eval_process = None
last_run = None
RESULTS_FILE = Path("eval/results.json")


@router.get("/results")
async def get_results() -> Dict[str, Any]:
    """Get RAGAS evaluation results from results.json."""
    if not RESULTS_FILE.exists():
        return {
            "metrics": {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0
            },
            "test_count": 0,
            "test_cases": [],
            "last_run": None
        }
    
    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
        
        # Extract average scores
        avg_scores = data.get("average_scores", {})
        test_cases = data.get("test_cases", [])
        scores = data.get("scores", [])
        
        # Build test cases with individual scores
        test_cases_with_scores = []
        for i, test_case in enumerate(test_cases):
            if i < len(scores):
                score_data = scores[i]
                faithfulness = score_data.get("faithfulness", 0.0)
                relevancy = score_data.get("answer_relevancy", 0.0)
                recall = score_data.get("context_recall", 0.0)
                
                # Determine if passed (all scores > 0.75)
                passed = faithfulness > 0.75 and relevancy > 0.75 and recall > 0.75
                
                test_cases_with_scores.append({
                    "question": test_case["question"],
                    "answer": test_case["answer"],
                    "ground_truth": test_case["ground_truth"],
                    "faithfulness_score": faithfulness,
                    "relevancy_score": relevancy,
                    "recall_score": recall,
                    "passed": passed
                })
            else:
                test_cases_with_scores.append({
                    "question": test_case["question"],
                    "answer": test_case["answer"],
                    "ground_truth": test_case["ground_truth"],
                    "faithfulness_score": 0.0,
                    "relevancy_score": 0.0,
                    "recall_score": 0.0,
                    "passed": False
                })
        
        # Get last run timestamp from file modification time
        last_run_timestamp = datetime.datetime.fromtimestamp(
            RESULTS_FILE.stat().st_mtime,
            tz=datetime.timezone.utc
        ).isoformat()
        
        return {
            "metrics": {
                "faithfulness": avg_scores.get("faithfulness", 0.0),
                "answer_relevancy": avg_scores.get("answer_relevancy", 0.0),
                "context_recall": avg_scores.get("context_recall", 0.0)
            },
            "test_count": len(test_cases),
            "test_cases": test_cases_with_scores,
            "last_run": last_run_timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {str(e)}")


@router.post("/run")
async def run_evaluation():
    """Trigger RAGAS evaluation subprocess."""
    global eval_status, eval_process
    
    if eval_status == "running":
        raise HTTPException(status_code=409, detail="Evaluation is already running")
    
    try:
        # Run evaluation in background
        eval_process = subprocess.Popen(
            ["uv", "run", "python", "eval/run_ragas.py"],
            cwd=".",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        eval_status = "running"
        
        return {"status": "running", "message": "Evaluation started"}
    except Exception as e:
        eval_status = "idle"
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get current evaluation status."""
    global eval_status, eval_process, last_run
    
    # Check if process is still running
    if eval_status == "running" and eval_process:
        if eval_process.poll() is not None:
            # Process has finished
            eval_status = "completed"
            eval_process = None
            # Update last run timestamp
            if RESULTS_FILE.exists():
                last_run = datetime.datetime.fromtimestamp(
                    RESULTS_FILE.stat().st_mtime,
                    tz=datetime.timezone.utc
                ).isoformat()
    
    return {
        "status": eval_status,
        "last_run": last_run
    }
