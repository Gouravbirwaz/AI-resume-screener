"""
Core Sentence-BERT Model and Semantic Matching Engine for Resume Screening.
Loads pretrained SentenceTransformer ('all-MiniLM-L6-v2') to generate dense
document embeddings and compute cosine similarity against Job Descriptions.
"""

import os
import sys

# Prevent transformers from loading TensorFlow/Keras and force pure PyTorch
sys.modules["tensorflow"] = None
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Allow immediate loading from local cache without network connection attempts
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import time
import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, util
from utils import extract_skills, compare_skills, extract_candidate_name, clean_text


class ResumeJobMatcher:
    """
    Sentence-BERT based semantic matcher for resumes and job descriptions.
    Uses 'all-MiniLM-L6-v2', a 6-layer MiniLM model fine-tuned on >1 billion sentence pairs
    for high semantic similarity performance with low latency (384-dim embeddings).
    """

    @staticmethod
    def resolve_model_path(model_name: str = "all-MiniLM-L6-v2") -> str:
        """Finds cached snapshot path if available to ensure zero-network offline loading."""
        user_home = os.path.expanduser("~")
        snapshots_dir = os.path.join(
            user_home, ".cache", "huggingface", "hub",
            f"models--sentence-transformers--{model_name}", "snapshots"
        )
        if os.path.exists(snapshots_dir):
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])
        return model_name

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        actual_path = self.resolve_model_path(model_name)
        print(f"Loading Sentence-BERT model from: {actual_path}...")
        start_t = time.time()
        self.model = SentenceTransformer(actual_path)
        self.load_time = round(time.time() - start_t, 3)
        print(f"Model loaded successfully in {self.load_time}s.")

    def compute_embedding(self, text: str) -> np.ndarray:
        """Generates 384-dimensional dense semantic embedding for input text."""
        cleaned = clean_text(text)
        if not cleaned:
            return np.zeros(384, dtype=np.float32)
        # S-BERT handles truncation and mean pooling automatically
        embedding = self.model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Computes cosine similarity between two normalized vectors:
        Cosine Sim = dot(A, B) / (||A|| * ||B||)
        Returns a percentage between 0.0% and 100.0%.
        """
        if np.all(vec_a == 0) or np.all(vec_b == 0):
            return 0.0
        # If normalized, dot product equals cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Clip to [-1, 1] for safety, then map to positive percentage
        similarity = max(0.0, min(1.0, similarity))
        return round(similarity * 100.0, 2)

    def evaluate_candidate(
        self,
        resume_text: str,
        jd_text: str,
        candidate_name: Optional[str] = None,
        semantic_weight: float = 0.70,
        skill_weight: float = 0.30
    ) -> Dict[str, Any]:
        """
        Deep evaluation of a candidate resume against a job description:
        1. Pure Semantic Similarity via Sentence-BERT
        2. Skill Overlap & Gap Analysis
        3. Weighted Composite Score
        4. Match Categorization & Qualitative Feedback
        """
        t0 = time.time()
        
        if not candidate_name:
            candidate_name = extract_candidate_name(resume_text)

        # 1. Compute S-BERT Embeddings
        resume_emb = self.compute_embedding(resume_text)
        jd_emb = self.compute_embedding(jd_text)

        # 2. Semantic Cosine Similarity
        semantic_score = self.compute_cosine_similarity(resume_emb, jd_emb)

        # 3. Skill Analysis
        skill_analysis = compare_skills(resume_text, jd_text)
        skill_overlap_ratio = skill_analysis["skill_overlap_ratio"]

        # 4. Composite Score (Semantic + Explicit Skill Overlap)
        composite_score = round(
            (semantic_score * semantic_weight) + (skill_overlap_ratio * skill_weight), 2
        )

        # 5. Categorize Match Tier
        if composite_score >= 75.0:
            tier = "Strong Match"
            recommendation = "Highly Recommended for Interview. Candidate demonstrates strong semantic and skill alignment."
            badge_color = "green"
        elif composite_score >= 55.0:
            tier = "Moderate Match"
            recommendation = "Recommended for Further Review. Relevant background with some skill gaps."
            badge_color = "orange"
        else:
            tier = "Low Match / Mismatched"
            recommendation = "Not Recommended. Candidate profile shows insufficient alignment with the job requirements."
            badge_color = "red"

        eval_latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "candidate_name": candidate_name,
            "semantic_score": semantic_score,
            "skill_score": skill_overlap_ratio,
            "composite_score": composite_score,
            "tier": tier,
            "badge_color": badge_color,
            "recommendation": recommendation,
            "matched_skills": skill_analysis["matched_skills"],
            "missing_skills": skill_analysis["missing_skills"],
            "extra_skills": skill_analysis["extra_skills"],
            "jd_skills_count": len(skill_analysis["jd_skills"]),
            "resume_skills_count": len(skill_analysis["resume_skills"]),
            "latency_ms": eval_latency_ms,
            "model_used": self.model_name
        }

    def batch_screen(
        self,
        resumes: List[Dict[str, str]],
        jd_text: str,
        semantic_weight: float = 0.70,
        skill_weight: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Screens a batch of candidate resumes against one Job Description and returns
        a ranked leaderboard sorted by composite match score descending.
        """
        results = []
        # Pre-compute JD embedding once to optimize batch processing
        jd_emb = self.compute_embedding(jd_text)
        jd_skills_info = extract_skills(jd_text)

        for res in resumes:
            text = res.get("text", "")
            name = res.get("name") or extract_candidate_name(text)
            filename = res.get("filename", "resume.txt")

            res_emb = self.compute_embedding(text)
            semantic_score = self.compute_cosine_similarity(res_emb, jd_emb)
            skill_info = compare_skills(text, jd_text)
            skill_score = skill_info["skill_overlap_ratio"]

            composite_score = round(
                (semantic_score * semantic_weight) + (skill_score * skill_weight), 2
            )

            if composite_score >= 75.0:
                tier = "Strong Match"
            elif composite_score >= 55.0:
                tier = "Moderate Match"
            else:
                tier = "Low Match"

            results.append({
                "candidate_name": name,
                "filename": filename,
                "composite_score": composite_score,
                "semantic_score": semantic_score,
                "skill_score": skill_score,
                "tier": tier,
                "matched_count": len(skill_info["matched_skills"]),
                "missing_count": len(skill_info["missing_skills"]),
                "matched_skills": skill_info["matched_skills"],
                "missing_skills": skill_info["missing_skills"]
            })

        # Rank candidates by composite score descending
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        return results
