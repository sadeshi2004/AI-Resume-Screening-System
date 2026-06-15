import os
import traceback
import streamlit as st
import pickle

st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.markdown("""
<style>
/* Page background and font */
body {background: linear-gradient(180deg, rgba(0, 19, 63, 0.75), rgba(7, 29, 74, 0.75)), url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1600&q=80'); background-size: cover; background-position: center; background-attachment: fixed;}

/* Make the Streamlit app container transparent so the page background shows through. */
div[data-testid="stAppViewContainer"], .stAppViewContainer, main .block-container, .main .block-container {
    background: rgba(255, 255, 255, 0.82) !important;
    backdrop-filter: blur(16px);
    box-shadow: 0 32px 90px rgba(15, 23, 42, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.55) !important;
}

.main {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #141b36; font-size: 21px;}

/* Header styles */
h1 {text-align: center; color: #0a4fff; font-size: 4.4rem; margin-bottom: 0.2rem; letter-spacing: -0.04em;}

/* Detail text */
p {font-size: 1.35rem; line-height: 1.95; color: #334155;}

/* Text area label */
.stTextArea label, .stTextArea > div > label, .stTextArea>div>div>label {
    font-size: 32px !important;
    font-weight: 700;
    color: #0f172a;
}

/* Wider text area */
.stTextArea>div>div>textarea {min-height: 320px; font-size: 23px; border-radius: 24px; padding: 26px; background: #f4f8ff; border: 1px solid #d8e4f4;}

/* Style predict button */
button.stButton>button {
    background-color: #0b5cff;
    color: white;
    border-radius: 18px;
    padding: 18px 32px;
    font-size: 1.2rem;
    font-weight: 700;
    box-shadow: 0 20px 40px rgba(11, 92, 255, 0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
button.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 24px 48px rgba(11, 92, 255, 0.30);
}

/* Prediction box */
.prediction-box {background: white; border-left: 8px solid #0b5cff; padding: 24px; border-radius:22px; box-shadow: 0 22px 52px rgba(15, 23, 42, 0.12);}
.prediction-box strong {font-size: 1.3rem;}

/* Section card */
.section-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), #eef5ff);
    border-radius: 34px;
    padding: 46px 48px;
    margin-bottom: 32px;
    box-shadow: 0 28px 80px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(15, 23, 42, 0.06);
}
.section-card:hover {
    transform: translateY(-1px);
    transition: transform 0.2s ease;
}

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, #f6fbff 0%, #eaf2ff 100%);
    border-radius: 32px;
    padding: 36px 42px;
    margin-bottom: 30px;
    box-shadow: 0 24px 62px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(15, 23, 42, 0.06);
}
.hero-card h2 {margin-top: 0; color: #0b4fff; font-size: 2.6rem;}
.hero-card p {margin: 0.5rem 0 0; color: #334155; font-size: 1.25rem;}

/* Sidebar styling */
section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #031f4b 0%, #0b3a77 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.12);
}
.sidebar-card {
    background: rgba(255, 255, 255, 0.08);
    color: #f8fafc;
    padding: 34px 26px;
    border-radius: 34px;
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.14);
}
.sidebar-card:hover {
    transform: translateY(-1px);
    transition: transform 0.2s ease;
}
.sidebar-header {font-size:26px; font-weight:700; color: #ffffff; margin-bottom: 12px}
.sidebar-sub {font-size:17px; color: #d0dff4; margin-bottom: 18px;}
.sidebar-badge {background: rgba(255, 255, 255, 0.12); color: #e2ecff; padding:12px 14px; border-radius:14px; font-size:15px; display:inline-block; margin-bottom:16px}
.sidebar-item {font-size: 17px; color: #e2e8f0; margin-bottom: 12px; line-height: 1.65;}
.file-list {font-size:16px; color:#e2e8f0; margin-top:14px; padding-left: 20px}
.file-list li {margin-bottom:8px}

div[data-testid="stAppViewContainer"], .main>div.block-container {
    background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.88)), url('https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=1600&q=80') !important;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
section.main>div.block-container {padding-top: 2.4rem; padding-bottom: 2.4rem;}
</style>
""", unsafe_allow_html=True)

st.title("AI Resume Screening System")
st.markdown("""
<div class='hero-card'>
  <h2>Clean, fast resume screening with AI</h2>
  <p>Paste in resume text and get a predicted job category in seconds.</p>
</div>
""", unsafe_allow_html=True)

# Diagnostic: show current working directory and files to help debug empty page issues
try:
    cwd = os.getcwd()
    files = sorted(os.listdir('.'))
except Exception:
    cwd, files = None, []

# Load model with status for sidebar
model = None
vectorizer = None
model_loaded = False
load_error = None
try:
    model = pickle.load(open("resume_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model_loaded = True
except Exception as e:
    load_error = traceback.format_exc()
    # Fallback dummy objects so the UI remains usable for testing
    class DummyVectorizer:
        def transform(self, X):
            return X

    class DummyModel:
        def predict(self, X):
            return ["UNKNOWN"] * len(X)

    vectorizer = DummyVectorizer()
    model = DummyModel()

# Sidebar: model status and diagnostics
with st.sidebar:
    # Build a styled sidebar using HTML
    files_html = ''
    if files:
        files_html = '<ul class="file-list">' + ''.join(f'<li>{f}</li>' for f in files) + '</ul>'

    status_html = ''
    if model_loaded:
        status_html = '<div class="sidebar-badge">Machine Learning Model Ready</div>'
    else:
        status_html = '<div class="sidebar-badge">Model: fallback (not loaded)</div>'

    sidebar_html = f"""
    <div class="sidebar-card">
        <div class="sidebar-header">AI Resume Screening</div>
        <div class="sidebar-sub">Enter resume text and get a job category prediction instantly.</div>
        {status_html}
        <div class='sidebar-item'>Professional screening in one place.</div>
        <div class='sidebar-item'>Use the text box to paste a resume and click Predict.</div>
    </div>
    """

    st.markdown(sidebar_html, unsafe_allow_html=True)
    if not model_loaded and load_error:
        st.info("Load error (first line):")
        st.text(load_error.splitlines()[0])

resume_text = st.text_area("Enter Resume Text", placeholder="e.g. Experienced Python developer with ML and NLP backgrounds...")

if st.button("Predict Category"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:
        text_vector = vectorizer.transform([resume_text])
        prediction = model.predict(text_vector)
        st.markdown(f"<div class='prediction-box'><strong>Predicted Job Category:</strong> {str(prediction[0])}</div>", unsafe_allow_html=True)

        if not model_loaded:
            st.info("Note: this is a fallback prediction (dummy model). Train and save a real model to enable real predictions.")
