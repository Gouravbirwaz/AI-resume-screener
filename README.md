# AI Resume Screening & Job Matching using Sentence-BERT
### NLP Mini Project — Application 5: Building Intelligent NLP Applications Using Pretrained Transformer Models

An end-to-end, production-grade Natural Language Processing (NLP) application that uses a pretrained **Sentence-BERT (Siamese Transformer)** model to evaluate, rank, and match candidate resumes against job descriptions based on deep semantic context and technical skill coverage.

---

## 🚀 Quick Start (How to Run)

### Option 1: One-Click Launcher (Windows)
Double-click the **`run.bat`** file in this folder. It will launch the Streamlit web application in your default browser at `http://localhost:8501`.

### Option 2: Terminal / Command Prompt
```bash
# 1. (Optional) Install dependencies if needed
pip install -r requirements.txt

# 2. Run the Streamlit Web Application
streamlit run app.py
```

### Option 3: Run the Automated Evaluation & Benchmark Script
```bash
python test_and_eval.py
```
This runs the full test suite comparing Sentence-BERT vs baseline TF-IDF across all test candidates, computes latency, and saves structured outputs into `evaluation_results.json`.

---

## 📂 Project Structure & Deliverables

| File / Folder | Purpose & Content |
|---|---|
| [PROJECT_REPORT.md](file:///c:/Users/Acer/Desktop/hod_assignment/PROJECT_REPORT.md) | **Complete Mini Project Report (Soft-Bind Ready)** covering all 7 mandatory sections, mathematical formulations, architecture diagrams, benchmark tables, and Viva preparation. |
| [app.py](file:///c:/Users/Acer/Desktop/hod_assignment/app.py) | **Streamlit Web Application** featuring single-candidate screening, skill gap badges, batch leaderboard ranking, interactive charts, and model theory inspector. |
| [matcher.py](file:///c:/Users/Acer/Desktop/hod_assignment/matcher.py) | **Sentence-BERT Core Engine** (`all-MiniLM-L6-v2`) for generating 384-D dense embeddings and calculating Cosine Similarity. |
| [utils.py](file:///c:/Users/Acer/Desktop/hod_assignment/utils.py) | PDF / TXT document parsers, text normalizer, and technical skill taxonomy extraction engine. |
| [test_and_eval.py](file:///c:/Users/Acer/Desktop/hod_assignment/test_and_eval.py) | Automated testing script benchmarking S-BERT against TF-IDF with latency and discrimination metrics. |
| [evaluation_results.json](file:///c:/Users/Acer/Desktop/hod_assignment/evaluation_results.json) | Empirical benchmark results, latency metrics, and test case records. |
| [run.bat](file:///c:/Users/Acer/Desktop/hod_assignment/run.bat) | Windows batch file to start the web app with a double-click. |
| [sample_data/](file:///c:/Users/Acer/Desktop/hod_assignment/sample_data) | Preloaded test resumes (Senior DS, Junior ML, Full-Stack Dev, Accountant) and Job Descriptions for instant demo. |

---

## 🧠 Transformer Model Details
- **Model Name:** `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture:** Siamese Transformer with Mean-Pooling
- **Embedding Dimensionality:** 384
- **Parameters:** ~22.7 Million
- **Distance Metric:** Cosine Similarity

---

## 👥 Mini Project Team Submission Details
- **Domain:** Human Resource Automation & Semantic Information Retrieval
- **Team Size:** 2 Students
- **Demonstration:** Refer to the Viva-Voce preparation guide in [PROJECT_REPORT.md](file:///c:/Users/Acer/Desktop/hod_assignment/PROJECT_REPORT.md#viva-voce--oral-defense-preparation-guide) for sample examiner questions and model answers.
