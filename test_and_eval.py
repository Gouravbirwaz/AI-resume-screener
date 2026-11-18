"""
Automated Testing and Performance Evaluation Script for Mini Project:
Resume Screening and Job Matching using Sentence-BERT.

Evaluates:
1. Sentence-BERT Semantic Cosine Similarity across multiple candidate resumes.
2. Comparison against baseline TF-IDF Lexical matching.
3. Inference latency and processing throughput (ms per candidate).
4. Ranking consistency across different job profiles.
5. Generates structured JSON and formatted markdown report of observations.
"""

import os
import sys

# Prevent transformers from loading TensorFlow/Keras and force pure PyTorch
sys.modules["tensorflow"] = None
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time
import json
import numpy as np
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine_similarity
from matcher import ResumeJobMatcher
from utils import extract_candidate_name, compare_skills

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JD_DIR = os.path.join(BASE_DIR, "sample_data", "job_descriptions")
RESUME_DIR = os.path.join(BASE_DIR, "sample_data", "resumes")


def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def compute_tfidf_similarity(text_a: str, text_b: str) -> float:
    """Baseline TF-IDF cosine similarity for comparative analysis."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    score = tfidf_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100.0, 2)


def run_comprehensive_evaluation():
    print("=" * 80)
    print("MINI PROJECT EVALUATION: RESUME SCREENING USING SENTENCE-BERT")
    print("=" * 80)

    # 1. Initialize Sentence-BERT Matcher
    t0 = time.time()
    matcher = ResumeJobMatcher("all-MiniLM-L6-v2")
    init_time = time.time() - t0
    print(f"[1/4] Sentence-BERT model loaded in {init_time:.2f} seconds.\n")

    # 2. Load Test Datasets
    jds = {
        "Data Scientist": load_file(os.path.join(JD_DIR, "data_scientist_jd.txt")),
        "Full-Stack Developer": load_file(os.path.join(JD_DIR, "fullstack_dev_jd.txt")),
        "DevOps Engineer": load_file(os.path.join(JD_DIR, "devops_engineer_jd.txt"))
    }

    resumes = [
        {"id": "cand_1", "file": "candidate_1_data_scientist_senior.txt", "label": "Senior Data Scientist (Aarav)"},
        {"id": "cand_2", "file": "candidate_2_ml_engineer_junior.txt", "label": "Junior ML Engineer (Priya)"},
        {"id": "cand_3", "file": "candidate_3_frontend_dev.txt", "label": "Full-Stack Dev (Rohan)"},
        {"id": "cand_4", "file": "candidate_4_accountant_finance.txt", "label": "Accountant / Finance (Sunita)"}
    ]

    for r in resumes:
        r["text"] = load_file(os.path.join(RESUME_DIR, r["file"]))

    print(f"[2/4] Loaded {len(jds)} Job Descriptions and {len(resumes)} Candidate Resumes.")

    # 3. Benchmark Evaluation across Scenarios
    evaluation_matrix = []
    latencies = []

    print("\n[3/4] Running Inference & Comparative Benchmark (Sentence-BERT vs TF-IDF Baseline)...")
    
    primary_jd_name = "Data Scientist"
    primary_jd_text = jds[primary_jd_name]

    print(f"\n--- Primary Target: {primary_jd_name} Position ---")
    print(f"{'Candidate':<32} | {'S-BERT':<8} | {'TF-IDF':<8} | {'Skills':<8} | {'Overall':<8} | {'Classification':<15}")
    print("-" * 92)

    for cand in resumes:
        t_start = time.time()
        eval_result = matcher.evaluate_candidate(
            resume_text=cand["text"],
            jd_text=primary_jd_text,
            candidate_name=cand["label"]
        )
        latency = (time.time() - t_start) * 1000
        latencies.append(latency)

        # Baseline TF-IDF computation
        tfidf_score = compute_tfidf_similarity(cand["text"], primary_jd_text)

        cand_eval = {
            "candidate_id": cand["id"],
            "candidate_name": cand["label"],
            "target_jd": primary_jd_name,
            "sbert_semantic_score": eval_result["semantic_score"],
            "tfidf_lexical_score": tfidf_score,
            "skill_match_score": eval_result["skill_score"],
            "composite_score": eval_result["composite_score"],
            "tier": eval_result["tier"],
            "matched_skills": eval_result["matched_skills"],
            "missing_skills": eval_result["missing_skills"],
            "latency_ms": round(latency, 2)
        }
        evaluation_matrix.append(cand_eval)

        print(
            f"{cand['label']:<32} | "
            f"{eval_result['semantic_score']:>6.1f}% | "
            f"{tfidf_score:>6.1f}% | "
            f"{eval_result['skill_score']:>6.1f}% | "
            f"{eval_result['composite_score']:>6.1f}% | "
            f"{eval_result['tier']:<15}"
        )

    # Cross-Job Validation: Check Full-Stack Candidate against Full-Stack JD
    print("\n--- Cross-Validation Target: Full-Stack Developer Position ---")
    print(f"{'Candidate':<32} | {'S-BERT':<8} | {'TF-IDF':<8} | {'Overall':<8} | {'Classification':<15}")
    print("-" * 80)
    fs_jd = jds["Full-Stack Developer"]
    cross_results = []
    for cand in resumes:
        eval_res = matcher.evaluate_candidate(cand["text"], fs_jd, candidate_name=cand["label"])
        tfidf_s = compute_tfidf_similarity(cand["text"], fs_jd)
        cross_results.append({
            "candidate": cand["label"],
            "sbert": eval_res["semantic_score"],
            "overall": eval_res["composite_score"],
            "tier": eval_res["tier"]
        })
        print(
            f"{cand['label']:<32} | "
            f"{eval_res['semantic_score']:>6.1f}% | "
            f"{tfidf_s:>6.1f}% | "
            f"{eval_res['composite_score']:>6.1f}% | "
            f"{eval_res['tier']:<15}"
        )

    # 4. Latency and Performance Summary
    avg_latency = np.mean(latencies)
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n[4/4] Performance & System Metrics:")
    print(f"  - Average Screening Latency: {avg_latency:.2f} ms per resume")
    print(f"  - Screening Throughput: ~{throughput:.1f} resumes / second (CPU)")
    print(f"  - Embedding Dimensionality: 384 dimensions")
    print(f"  - Vector Distance Metric: Cosine Similarity")

    # Save to JSON
    output_summary = {
        "project_title": "Resume Screening and Job Matching using Sentence-BERT",
        "model_architecture": "all-MiniLM-L6-v2 (Siamese Transformer)",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_summary": {
            "avg_latency_ms": round(float(avg_latency), 2),
            "throughput_resumes_per_sec": round(float(throughput), 1),
            "embedding_dim": 384
        },
        "primary_jd_evaluation": evaluation_matrix,
        "cross_validation_evaluation": cross_results
    }

    results_file = os.path.join(BASE_DIR, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=4)
    print(f"\nSaved structured evaluation report to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_evaluation()
