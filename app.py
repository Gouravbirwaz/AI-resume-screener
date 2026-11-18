"""
Streamlit Web Application: Resume Screening and Job Matching using Sentence-BERT.
Academic Mini Project Demonstration with Interactive Evaluation, Skill Extraction,
Visualizations, and Batch Candidate Ranking.
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

import streamlit as st
import pandas as pd
import numpy as np
from matcher import ResumeJobMatcher
from utils import extract_text_from_pdf, clean_text, extract_candidate_name

# Set page layout and title
st.set_page_config(
    page_title="AI Resume Screener (Sentence-BERT)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #2563EB;
    }
    .skill-tag-match {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 16px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-tag-missing {
        display: inline-block;
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 16px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Sentence-BERT Transformer Model...")
def load_matcher():
    """Caches the model in memory so it only loads once."""
    return ResumeJobMatcher(model_name="all-MiniLM-L6-v2")


matcher = load_matcher()

# Sample data directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_JD_DIR = os.path.join(BASE_DIR, "sample_data", "job_descriptions")
SAMPLE_RESUME_DIR = os.path.join(BASE_DIR, "sample_data", "resumes")


def get_sample_files(directory):
    if os.path.exists(directory):
        return [f for f in os.listdir(directory) if f.endswith(".txt")]
    return []


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
st.sidebar.title("Configuration")
st.sidebar.markdown("**Mini Project: Application 5**\n*Sentence-BERT NLP Screener*")

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Scoring Weights")
semantic_weight = st.sidebar.slider(
    "Sentence-BERT Semantic Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
    help="Weight given to contextual deep semantic similarity derived from Sentence-BERT embeddings."
)
skill_weight = round(1.0 - semantic_weight, 2)
st.sidebar.markdown(f"**Skill Overlap Weight:** `{skill_weight:.2f}`")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Load Demo Data")
sample_jds = get_sample_files(SAMPLE_JD_DIR)
sample_resumes = get_sample_files(SAMPLE_RESUME_DIR)

selected_sample_jd = st.sidebar.selectbox("Choose Sample Job Description", ["None"] + sample_jds)
selected_sample_res = st.sidebar.selectbox("Choose Sample Resume", ["None"] + sample_resumes)

st.sidebar.markdown("---")
st.sidebar.info("""
**Pretrained Model Details:**
- **Architecture:** Siamese MiniLM (Sentence-BERT)
- **Base Model:** `all-MiniLM-L6-v2`
- **Embedding Dimension:** 384 dense floats
- **Pooling Strategy:** Mean Pooling
- **Distance Metric:** Cosine Similarity
""")

# App Header
st.markdown('<div class="main-header">🎯 AI Resume Screening & Job Matching System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Built with Pretrained <b>Sentence-BERT (Siamese Transformer)</b> | NLP Mini Project</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Candidate Screening",
    "📊 Batch Screening & Ranking",
    "🧠 Sentence-BERT Architecture",
    "📖 Mini Project Details & Viva Prep"
])

# ============================================================================
# TAB 1: SINGLE CANDIDATE SCREENING
# ============================================================================
with tab1:
    col_left, col_right = st.columns(2)

    # Pre-fill JD if sample selected
    default_jd_text = ""
    if selected_sample_jd != "None":
        with open(os.path.join(SAMPLE_JD_DIR, selected_sample_jd), "r", encoding="utf-8") as f:
            default_jd_text = f.read()

    # Pre-fill Resume if sample selected
    default_res_text = ""
    if selected_sample_res != "None":
        with open(os.path.join(SAMPLE_RESUME_DIR, selected_sample_res), "r", encoding="utf-8") as f:
            default_res_text = f.read()

    with col_left:
        st.subheader("1. Job Description (JD)")
        jd_input = st.text_area(
            "Paste or edit Job Description:",
            value=default_jd_text,
            height=280,
            placeholder="Paste role requirements, responsibilities, and required technical skills..."
        )

    with col_right:
        st.subheader("2. Candidate Resume")
        uploaded_resume = st.file_uploader("Upload Resume (.pdf, .txt)", type=["pdf", "txt"])

        if uploaded_resume is not None:
            if uploaded_resume.name.endswith(".pdf"):
                resume_input = extract_text_from_pdf(uploaded_resume.read())
            else:
                resume_input = uploaded_resume.read().decode("utf-8", errors="ignore")
        else:
            resume_input = st.text_area(
                "Or paste Resume text below:",
                value=default_res_text,
                height=240,
                placeholder="Paste candidate resume, work history, skills, and education..."
            )

    screen_btn = st.button("🚀 Analyze & Match Resume", type="primary", use_container_width=True)

    if screen_btn:
        if not jd_input.strip() or not resume_input.strip():
            st.error("⚠️ Please provide both a Job Description and a Candidate Resume to evaluate.")
        else:
            with st.spinner("Analyzing semantic embeddings and skill alignment..."):
                eval_res = matcher.evaluate_candidate(
                    resume_text=resume_input,
                    jd_text=jd_input,
                    semantic_weight=semantic_weight,
                    skill_weight=skill_weight
                )

            st.markdown("---")
            st.subheader(f"📋 Screening Results: {eval_res['candidate_name']}")

            # Score Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Overall Match Score", f"{eval_res['composite_score']}%")
            with m2:
                st.metric("Sentence-BERT Semantic", f"{eval_res['semantic_score']}%")
            with m3:
                st.metric("Skill Coverage", f"{eval_res['skill_score']}%")
            with m4:
                tier = eval_res["tier"]
                if tier == "Strong Match":
                    st.success(f"**{tier}**")
                elif tier == "Moderate Match":
                    st.warning(f"**{tier}**")
                else:
                    st.error(f"**{tier}**")

            # Progress Bar
            st.progress(eval_res["composite_score"] / 100.0)
            st.info(f"💡 **Recommendation:** {eval_res['recommendation']}")

            # Skills Gap Breakdown
            st.markdown("#### 🛠️ Detailed Skill Alignment Breakdown")
            c_skill1, c_skill2 = st.columns(2)

            with c_skill1:
                st.markdown(f"**✅ Matched Skills ({len(eval_res['matched_skills'])}):**")
                if eval_res["matched_skills"]:
                    tags_html = "".join([f'<span class="skill-tag-match">{s.title()}</span>' for s in eval_res["matched_skills"]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*No direct skill keywords overlapped.*")

            with c_skill2:
                st.markdown(f"**⚠️ Missing / Desired Skills ({len(eval_res['missing_skills'])}):**")
                if eval_res["missing_skills"]:
                    tags_html = "".join([f'<span class="skill-tag-missing">{s.title()}</span>' for s in eval_res["missing_skills"]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*Candidate covers all key skills identified in the Job Description!*")

            # Additional skills
            if eval_res["extra_skills"]:
                with st.expander(f"➕ Candidate's Additional Skills ({len(eval_res['extra_skills'])})"):
                    extra_tags = " ".join([f"`{s.title()}`" for s in eval_res["extra_skills"]])
                    st.markdown(extra_tags)

            st.caption(f"⚡ Evaluation completed in {eval_res['latency_ms']} ms using Sentence-BERT `{eval_res['model_used']}`.")

# ============================================================================
# TAB 2: BATCH SCREENING & CANDIDATE RANKING
# ============================================================================
with tab2:
    st.subheader("👥 Batch Candidate Screening & Leaderboard")
    st.markdown("Upload multiple candidate resumes or use preloaded samples to screen and rank candidates simultaneously.")

    batch_jd_text = default_jd_text if default_jd_text else ""
    batch_jd = st.text_area("Job Description for Batch Ranking:", value=batch_jd_text, height=160)

    batch_files = st.file_uploader(
        "Upload Multiple Resumes (PDF / TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    use_sample_resumes = st.checkbox("Include all 4 preloaded sample resumes in batch evaluation", value=True)

    if st.button("📊 Run Batch Screening", type="primary"):
        if not batch_jd.strip():
            st.error("Please provide a Job Description.")
        else:
            resumes_to_screen = []

            # 1. Add uploaded files
            if batch_files:
                for bf in batch_files:
                    if bf.name.endswith(".pdf"):
                        txt = extract_text_from_pdf(bf.read())
                    else:
                        txt = bf.read().decode("utf-8", errors="ignore")
                    resumes_to_screen.append({
                        "filename": bf.name,
                        "text": txt,
                        "name": extract_candidate_name(txt)
                    })

            # 2. Add sample resumes if requested
            if use_sample_resumes and os.path.exists(SAMPLE_RESUME_DIR):
                for s_res in os.listdir(SAMPLE_RESUME_DIR):
                    if s_res.endswith(".txt"):
                        with open(os.path.join(SAMPLE_RESUME_DIR, s_res), "r", encoding="utf-8") as f:
                            txt = f.read()
                        resumes_to_screen.append({
                            "filename": s_res,
                            "text": txt,
                            "name": extract_candidate_name(txt)
                        })

            if not resumes_to_screen:
                st.warning("No resumes provided. Please upload resumes or check 'Include sample resumes'.")
            else:
                with st.spinner(f"Screening and ranking {len(resumes_to_screen)} candidates using Sentence-BERT..."):
                    ranked_candidates = matcher.batch_screen(
                        resumes=resumes_to_screen,
                        jd_text=batch_jd,
                        semantic_weight=semantic_weight,
                        skill_weight=skill_weight
                    )

                st.success(f"Successfully evaluated and ranked {len(ranked_candidates)} candidates!")

                # Convert to DataFrame
                df_data = []
                for c in ranked_candidates:
                    df_data.append({
                        "Rank": c["rank"],
                        "Candidate Name": c["candidate_name"],
                        "File Name": c["filename"],
                        "Overall Score (%)": c["composite_score"],
                        "S-BERT Semantic (%)": c["semantic_score"],
                        "Skill Match (%)": c["skill_score"],
                        "Matched Skills Count": c["matched_count"],
                        "Missing Skills Count": c["missing_count"],
                        "Tier": c["tier"]
                    })
                df = pd.DataFrame(df_data)

                # Display interactive leaderboard
                st.dataframe(df, use_container_width=True)

                # Visualization: Score Comparison Bar Chart
                st.markdown("#### 📈 Candidate Score Comparison")
                chart_df = df.set_index("Candidate Name")[["Overall Score (%)", "S-BERT Semantic (%)", "Skill Match (%)"]]
                st.bar_chart(chart_df)

                # Download CSV button
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Screening Report (CSV)",
                    data=csv_bytes,
                    file_name="resume_screening_report.csv",
                    mime="text/csv"
                )

# ============================================================================
# TAB 3: SENTENCE-BERT ARCHITECTURE & THEORY
# ============================================================================
with tab3:
    st.subheader("🧠 Pretrained Transformer Model: Sentence-BERT Architecture")
    
    st.markdown("""
    ### Why Sentence-BERT (SBERT) instead of Vanilla BERT?
    Standard BERT uses a cross-encoder architecture where pairs of sentences are concatenated with a `[SEP]` token: `[CLS] Resume [SEP] Job Description [SEP]`.
    - **Vanilla BERT Bottleneck:** To find the best matching resume among $N$ candidates against $M$ jobs requires $N \\times M$ full forward passes through BERT. For 1,000 resumes and 10 job descriptions, BERT requires 10,000 cross-encoding computations, which takes **hours**!
    - **Sentence-BERT Breakthrough:** Reimers & Gurevych (EMNLP 2019) introduced Siamese and Triplet network structures that derive fixed-size, semantically meaningful sentence embeddings ($u$ and $v$).
    - **Inference Speedup:** Embeddings are pre-computed once ($O(N)$), and similarity is computed using **Cosine Similarity**, completing screening in **milliseconds** (over **1000x faster**).
    """)

    st.markdown("---")
    st.markdown("""
    ### Mathematical Formulation
    
    1. **Sentence Embedding Generation:**
       $$\\mathbf{u} = \\text{Mean-Pooling}(\\text{BERT}(\\text{Resume}))$$
       $$\\mathbf{v} = \\text{Mean-Pooling}(\\text{BERT}(\\text{Job Description}))$$
       
       Where Mean-Pooling calculates the average of all contextual token vectors generated by the final transformer layer, taking the attention mask into account.
    
    2. **Cosine Similarity Computation:**
       $$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2} = \\frac{\\sum_{i=1}^{d} u_i v_i}{\\sqrt{\\sum_{i=1}^{d} u_i^2} \\sqrt{\\sum_{i=1}^{d} v_i^2}}$$
       Where $d = 384$ for `all-MiniLM-L6-v2`.
       
    3. **Composite Match Scoring:**
       $$\\text{Match Score} = (w_{\\text{semantic}} \\times \\text{Cosine Sim}) + (w_{\\text{skills}} \\times \\text{Skill Overlap Ratio})$$
    """)

    st.markdown("---")
    st.markdown("""
    ### Transformer Model Specifications (`all-MiniLM-L6-v2`)
    - **Base Architecture:** 6-layer Transformer Encoder (MiniLM distillation)
    - **Hidden Dimension:** 384
    - **Attention Heads:** 12
    - **Parameters:** ~22.7 Million (lightweight and ultra-fast)
    - **Pre-training Objective:** Contrastive learning over 1 Billion sentence pairs
    - **Speed:** ~14,000 sentences/second on GPU, ~2,000 sentences/second on modern multi-core CPU.
    """)

# ============================================================================
# TAB 4: MINI PROJECT DETAILS & VIVA PREPARATION
# ============================================================================
with tab4:
    st.subheader("🎓 Mini Project Submission & Viva-Voce Defense Guide")
    st.markdown("""
    **Project Title:** Building Intelligent NLP Applications Using Pretrained Transformer Models  
    **Chosen Application (Option 5):** Resume Screening and Job Matching using Sentence-BERT  
    **Evaluation Checklist:**
    - [x] Identified real-world problem and defined objectives.
    - [x] Selected appropriate pretrained Transformer model (`all-MiniLM-L6-v2`).
    - [x] Loaded pretrained model and tokenizer using Python (`sentence-transformers`).
    - [x] Developed interactive user interface using Streamlit.
    - [x] Accepted user input (PDF/text upload & text input) and generated predictions.
    - [x] Tested application using multiple test cases (Data Scientist, ML, Full-Stack, Accountant).
    - [x] Evaluated performance metrics (Semantic Cosine Sim, Skill Match, Latency).
    - [x] Prepared soft-bind compliant Mini Project Report & demonstration scripts.
    """)

    st.markdown("---")
    st.subheader("💡 Frequently Asked Viva Questions (For Team Members)")

    with st.expander("Q1: What is Sentence-BERT and how does it differ from standard BERT?"):
        st.write("""
        **Answer:** Standard BERT takes pairs of sentences concatenated together through cross-attention, which does not output an independent sentence embedding and requires $O(n^2)$ pairwise passes. Sentence-BERT uses a Siamese/Triplet network architecture to project each document independently into a 384-dimensional dense semantic vector space. We can then compare documents using simple vector operations like Cosine Similarity in constant time.
        """)

    with st.expander("Q2: Why is Mean Pooling used instead of using the [CLS] token?"):
        st.write("""
        **Answer:** Studies by Reimers & Gurevych showed that BERT's native `[CLS]` token embedding often yields poor sentence representations for semantic similarity (often performing worse than simple GloVe embeddings). Mean Pooling averages the contextual token representations across the sequence, capturing the global semantic meaning of the document far more reliably.
        """)

    with st.expander("Q3: How does your system prevent keyword stuffing?"):
        st.write("""
        **Answer:** Legacy Applicant Tracking Systems (ATS) use simple keyword counting (TF-IDF or regex), which candidates exploit by pasting invisible keyword lists. Our system uses Sentence-BERT semantic embeddings which evaluate the **contextual meaning** of phrases and experience sentences. A candidate who simply lists keywords without coherent context will score low on deep semantic alignment.
        """)

    with st.expander("Q4: What performance metrics did you use to evaluate the system?"):
        st.write("""
        **Answer:** 
        1. **Cosine Similarity:** Measures directional angle between normalized 384-D document vectors ($0$ to $100\\%$).
        2. **Skill Overlap Ratio (Jaccard-like):** Measures exact match of recognized technical competencies.
        3. **Inference Latency:** Milliseconds taken to encode and score resumes.
        4. **Ranking Discrimination:** Ability of the system to clearly separate strong matches ($>80\\%$) from moderate matches ($50-70\\%$) and completely unrelated candidates ($<35\\%$).
        """)
