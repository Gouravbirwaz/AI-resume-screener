"""
Utility functions for Resume Screening and Job Matching:
- PDF / TXT document parsing
- Text normalization & preprocessing
- Technology & domain skill extraction
- Section segmentation
"""

import re
from typing import List, Set, Dict, Tuple, Optional


# Pre-defined comprehensive skill taxonomy across tech and business domains
TAXONOMY_SKILLS = {
    # Programming Languages
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "r", "go", "golang",
    "rust", "ruby", "php", "scala", "kotlin", "swift", "sql", "bash", "shell",
    
    # Machine Learning & AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "transformers", "bert", "sentence-bert", "sbert", "roberta",
    "gpt", "gpt-2", "llm", "large language models", "hugging face", "pytorch",
    "tensorflow", "keras", "scikit-learn", "opencv", "spacy", "nltk", "langchain",
    "reinforcement learning", "yolo", "xgboost", "lightgbm", "data science",
    "data analysis", "statistics", "feature engineering", "mlops", "mlflow",
    
    # Data & Analytics
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "spark", "apache spark",
    "hadoop", "kafka", "tableau", "power bi", "bigquery", "snowflake", "etl",
    "data warehousing",
    
    # Web & Software Development
    "react", "react.js", "angular", "vue", "vue.js", "next.js", "node.js", "nodejs",
    "express", "express.js", "fastapi", "flask", "django", "spring boot", "html",
    "html5", "css", "css3", "tailwind", "tailwind css", "bootstrap", "graphql",
    "rest api", "restful api", "microservices", "web development",
    
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
    "cassandra", "dynamodb", "oracle",
    
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
    "kubernetes", "k8s", "terraform", "ansible", "jenkins", "git", "github",
    "gitlab", "ci/cd", "prometheus", "grafana", "linux", "unix", "sre",
    
    # Finance, Business & Soft Skills
    "tally", "sap", "gst", "tds", "audit", "statutory audit", "taxation",
    "financial analysis", "excel", "financial modeling", "budgeting",
    "agile", "scrum", "jira", "communication", "leadership", "project management"
}


def clean_text(text: str) -> str:
    """Cleans and standardizes raw text extracted from documents."""
    if not text:
        return ""
    # Normalize bullet points and dashes
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\uf0a7\u25aa\u25cf]", "\n- ", text)
    # Remove control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace excessive consecutive newlines with two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_bytes_or_path) -> str:
    """Extracts text from PDF bytes or filepath using pypdf/pymupdf fallback."""
    text_content = []
    
    # Method 1: Try PyMuPDF (fitz) if installed for high fidelity
    try:
        import fitz  # PyMuPDF
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            doc = fitz.open(stream=file_bytes_or_path, filetype="pdf")
        else:
            doc = fitz.open(file_bytes_or_path)
            
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text_content.append(page_text)
        doc.close()
        combined = "\n".join(text_content)
        if combined.strip():
            return clean_text(combined)
    except Exception:
        pass
    
    # Method 2: Try pypdf fallback
    try:
        import io
        from pypdf import PdfReader
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            reader = PdfReader(io.BytesIO(file_bytes_or_path))
        else:
            reader = PdfReader(file_bytes_or_path)
            
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_content.append(t)
        return clean_text("\n".join(text_content))
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


def extract_skills(text: str) -> List[str]:
    """
    Extracts recognized technical and domain skills from document text using regex
    word-boundary matching against the comprehensive skill taxonomy.
    """
    if not text:
        return []
        
    lower_text = f" {text.lower()} "
    found_skills = set()
    
    for skill in TAXONOMY_SKILLS:
        # Escape special regex chars like ++, #, .
        escaped_skill = re.escape(skill)
        pattern = rf"(?<!\w){escaped_skill}(?!\w)"
        if re.search(pattern, lower_text):
            found_skills.add(skill)
            
    # Normalize aliases (e.g. react.js -> react, postgres -> postgresql)
    alias_map = {
        "react.js": "react",
        "vue.js": "vue",
        "node.js": "node.js",
        "nodejs": "node.js",
        "express.js": "express",
        "tailwind css": "tailwind",
        "k8s": "kubernetes",
        "sbert": "sentence-bert",
        "postgres": "postgresql",
        "golang": "go"
    }
    
    canonical_skills = set()
    for s in found_skills:
        canonical_skills.add(alias_map.get(s, s))
        
    return sorted(list(canonical_skills))


def compare_skills(resume_text: str, jd_text: str) -> Dict[str, any]:
    """
    Identifies matched, missing, and extra skills between candidate and job description.
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))
    
    matched = sorted(list(resume_skills.intersection(jd_skills)))
    missing = sorted(list(jd_skills - resume_skills))
    extra = sorted(list(resume_skills - jd_skills))
    
    overlap_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0
    
    return {
        "jd_skills": sorted(list(jd_skills)),
        "resume_skills": sorted(list(resume_skills)),
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "skill_overlap_ratio": round(overlap_ratio * 100, 2)
    }


def extract_candidate_name(text: str) -> str:
    """Extracts probable candidate name from resume headers."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return "Unknown Candidate"
    # Usually candidate name is on the first 1-2 non-empty lines
    first_line = lines[0]
    if len(first_line.split()) <= 4 and not re.search(r"@|resume|curriculum|phone|email", first_line, re.IGNORECASE):
        return first_line
    return lines[0][:40]
