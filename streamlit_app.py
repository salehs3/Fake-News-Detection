import streamlit as st
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Page config ──
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Serif+4&display=swap');

    .masthead {
        text-align: center;
        padding: 1rem 0 0.5rem;
        border-bottom: 3px solid #1a1a1a;
        margin-bottom: 1.5rem;
    }
    .masthead h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin: 0;
        color: #1a1a1a;
    }
    .masthead h1 span { color: #c0392b; }
    .masthead-sub {
        font-size: 11px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #6b6b6b;
        margin-top: 0.2rem;
    }
    .politics-banner {
        background: #3C3B6E;
        color: white;
        text-align: center;
        padding: 0.4rem;
        font-size: 12px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    .politics-banner strong { color: #FFD700; }

    .verdict-fake {
        background: #f9eae9;
        border-left: 4px solid #c0392b;
        padding: 1rem 1.2rem;
        border-radius: 0 4px 4px 0;
    }
    .verdict-real {
        background: #e8f5ee;
        border-left: 4px solid #1a6b3c;
        padding: 1rem 1.2rem;
        border-radius: 0 4px 4px 0;
    }
    .verdict-title-fake {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #c0392b;
    }
    .verdict-title-real {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a6b3c;
    }
    .section-label {
        font-size: 10px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #c0392b;
        font-weight: 600;
        border-top: 2px solid #c0392b;
        padding-top: 0.3rem;
        margin-bottom: 0.8rem;
    }
    .breakdown-bert {
        background: #f5eef8;
        border: 1px solid #7b2d8b;
        padding: 0.8rem;
        border-radius: 4px;
    }
    .breakdown-tfidf {
        background: #eaf2fb;
        border: 1px solid #1a5276;
        padding: 0.8rem;
        border-radius: 4px;
    }
    .word-fake { color: #c0392b; font-weight: 600; }
    .word-real { color: #1a6b3c; font-weight: 600; }
    .disclaimer {
        font-size: 11px;
        color: #6b6b6b;
        font-style: italic;
        text-align: center;
        margin-top: 0.5rem;
    }
    .history-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e8e3da;
        font-size: 13px;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = 'model'

# ── Load models ──
@st.cache_resource
def load_all_models():
    import os
    models = {}
    files = ['tfidf_model.pkl', 'bert_only_model.pkl', 'meta_model.pkl',
             'vectorizer.pkl', 'scaler.pkl']
    for f in files:
        path = os.path.join(MODEL_DIR, f)
        if not os.path.exists(path):
            return None, f"Missing: {f}. Run train_combined.py first."
        with open(path, 'rb') as fp:
            models[f.replace('.pkl', '')] = pickle.load(fp)

    info_path = os.path.join(MODEL_DIR, 'model_info.pkl')
    if os.path.exists(info_path):
        with open(info_path, 'rb') as f:
            models['info'] = pickle.load(f)
    else:
        models['info'] = {}

    bert_name = models['info'].get('bert_model_name', 'all-MiniLM-L6-v2')
    models['bert'] = SentenceTransformer(bert_name)
    return models, None

# ── Helper functions ──
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join([w for w in text.split() if len(w) > 2])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text, models):
    cleaned = clean_text(text)
    vectorized = models['vectorizer'].transform([cleaned])
    tfidf_proba = models['tfidf_model'].predict_proba(vectorized)

    bert_text = text[:1500]
    bert_emb = models['bert'].encode([bert_text])
    bert_scaled = models['scaler'].transform(bert_emb)
    bert_proba = models['bert_only_model'].predict_proba(bert_scaled)

    meta_features = np.hstack([tfidf_proba, bert_proba])
    final_proba = models['meta_model'].predict_proba(meta_features)[0]
    final_pred = models['meta_model'].predict(meta_features)[0]

    return {
        'prediction': int(final_pred),
        'fake_pct': round(float(final_proba[0]) * 100, 1),
        'real_pct': round(float(final_proba[1]) * 100, 1),
        'tfidf_fake_pct': round(float(tfidf_proba[0][0]) * 100, 1),
        'tfidf_real_pct': round(float(tfidf_proba[0][1]) * 100, 1),
        'bert_fake_pct': round(float(bert_proba[0][0]) * 100, 1),
        'bert_real_pct': round(float(bert_proba[0][1]) * 100, 1),
    }

def get_word_importance(text, models, n=6):
    cleaned = clean_text(text)
    vectorized = models['vectorizer'].transform([cleaned])
    feature_names = models['vectorizer'].get_feature_names_out()
    coefficients = models['tfidf_model'].coef_[0]
    article_features = vectorized.toarray()[0]
    word_importance = article_features * coefficients

    present_indices = np.where(article_features > 0)[0]
    if len(present_indices) == 0:
        return [], []

    present_importance = word_importance[present_indices]
    present_words = feature_names[present_indices]
    sorted_idx = np.argsort(present_importance)

    fake_words = [
        (present_words[i], round(abs(float(present_importance[i])) * 100, 1))
        for i in sorted_idx[:n] if present_importance[i] < 0
    ]
    real_words = [
        (present_words[i], round(abs(float(present_importance[i])) * 100, 1))
        for i in sorted_idx[-n:][::-1] if present_importance[i] > 0
    ]
    return sorted(fake_words, key=lambda x: x[1], reverse=True), \
           sorted(real_words, key=lambda x: x[1], reverse=True)

# ── Initialize session state ──
if 'history' not in st.session_state:
    st.session_state.history = []

# ── MASTHEAD ──
st.markdown("""
<div class="masthead">
    <h1>Truth<span>Lens</span></h1>
    <div class="masthead-sub">AI-Powered News Credibility Analyzer · Hybrid TF-IDF + BERT</div>
</div>
<div class="politics-banner">
    🇺🇸 Specialized for <strong>American Political News</strong> — Check Your Politics Headlines Here 🇺🇸
</div>
""", unsafe_allow_html=True)

# ── Load models ──
with st.spinner("Loading models — this may take 30 seconds on first load..."):
    models, error = load_all_models()

if error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ── Layout ──
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown('<div class="section-label">Submit Article for Analysis</div>', unsafe_allow_html=True)

    st.info("🗳 Best for U.S. political news — elections, Congress, White House, political figures. Results may be unreliable for sports, science, or entertainment.")

    article_text = st.text_area(
        "Paste the full text of a U.S. political news article",
        height=220,
        placeholder="Paste article text here...",
        label_visibility="collapsed"
    )

    col_chars, col_words = st.columns(2)
    if article_text:
        col_chars.caption(f"{len(article_text):,} characters")
        col_words.caption(f"{len(article_text.split()):,} words")

    analyze_btn = st.button("▶ Analyze Article", type="primary", use_container_width=True)
    st.markdown('<p class="disclaimer">⚠ Detects writing style patterns, not factual accuracy. Always verify with trusted sources.</p>', unsafe_allow_html=True)

    # ── Analysis ──
    if analyze_btn:
        if len(article_text.strip()) < 50:
            st.error("Please enter at least 50 characters for a reliable result.")
        else:
            with st.spinner("Running TF-IDF · Running BERT · Combining predictions..."):
                result = predict(article_text, models)
                fake_words, real_words = get_word_importance(article_text, models)

            verdict = 'FAKE' if result['prediction'] == 0 else 'REAL'
            confidence = result['fake_pct'] if verdict == 'FAKE' else result['real_pct']
            is_fake = verdict == 'FAKE'

            # ── Verdict ──
            st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

            if is_fake:
                st.markdown(f"""
                <div class="verdict-fake">
                    <div class="verdict-title-fake">⚠ Likely Fake News</div>
                    <div style="font-size:13px;color:#6b6b6b;margin-top:0.3rem;">
                        {confidence}% confidence · {len(article_text.split()):,} words · Hybrid TF-IDF + BERT
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-real">
                    <div class="verdict-title-real">✓ Likely Credible</div>
                    <div style="font-size:13px;color:#6b6b6b;margin-top:0.3rem;">
                        {confidence}% confidence · {len(article_text.split()):,} words · Hybrid TF-IDF + BERT
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Model breakdown ──
            st.markdown('<div class="section-label">Model Breakdown</div>', unsafe_allow_html=True)
            bc1, bc2 = st.columns(2)

            tfidf_verdict = "Likely Fake" if result['tfidf_fake_pct'] > 50 else "Likely Real"
            tfidf_conf = result['tfidf_fake_pct'] if result['tfidf_fake_pct'] > 50 else result['tfidf_real_pct']
            bert_verdict = "Likely Fake" if result['bert_fake_pct'] > 50 else "Likely Real"
            bert_conf = result['bert_fake_pct'] if result['bert_fake_pct'] > 50 else result['bert_real_pct']

            with bc1:
                st.markdown(f"""
                <div class="breakdown-tfidf">
                    <div style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#1a5276;font-weight:700;">TF-IDF Keywords</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{'#c0392b' if 'Fake' in tfidf_verdict else '#1a6b3c'}">{tfidf_verdict}</div>
                    <div style="font-size:12px;color:#6b6b6b;">{tfidf_conf}% confidence</div>
                    <div style="font-size:11px;color:#6b6b6b;font-style:italic;">Keyword frequency patterns</div>
                </div>""", unsafe_allow_html=True)

            with bc2:
                st.markdown(f"""
                <div class="breakdown-bert">
                    <div style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#7b2d8b;font-weight:700;">BERT Semantics</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{'#c0392b' if 'Fake' in bert_verdict else '#1a6b3c'}">{bert_verdict}</div>
                    <div style="font-size:12px;color:#6b6b6b;">{bert_conf}% confidence</div>
                    <div style="font-size:11px;color:#6b6b6b;font-style:italic;">Deep language understanding</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Confidence bars ──
            st.markdown('<div class="section-label">Final Hybrid Confidence</div>', unsafe_allow_html=True)
            pc1, pc2 = st.columns(2)
            with pc1:
                st.caption(f"Fake Probability — {result['fake_pct']}%")
                st.progress(result['fake_pct'] / 100)
            with pc2:
                st.caption(f"Real Probability — {result['real_pct']}%")
                st.progress(result['real_pct'] / 100)

            # ── Word importance ──
            st.markdown('<div class="section-label">Word Importance Analysis (TF-IDF Component)</div>', unsafe_allow_html=True)
            wc1, wc2 = st.columns(2)

            with wc1:
                st.markdown("**⚠ Fake Indicators**")
                if fake_words:
                    max_score = fake_words[0][1]
                    for word, score in fake_words:
                        bar = int((score / max_score) * 20)
                        st.markdown(f"`{word}` {'█' * bar} {score}")
                else:
                    st.caption("No significant fake indicators found")

            with wc2:
                st.markdown("**✓ Real Indicators**")
                if real_words:
                    max_score = real_words[0][1]
                    for word, score in real_words:
                        bar = int((score / max_score) * 20)
                        st.markdown(f"`{word}` {'█' * bar} {score}")
                else:
                    st.caption("No significant real indicators found")

            st.caption("Hybrid model: Logistic Regression on TF-IDF + BERT (all-MiniLM-L6-v2) · Meta-classifier combines both · Trained on 44,898 U.S. political news articles")

            # ── Add to history ──
            st.session_state.history.insert(0, {
                'text': article_text[:120] + ('...' if len(article_text) > 120 else ''),
                'verdict': verdict,
                'confidence': confidence
            })

    # ── History ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Analysis History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.caption("No articles analyzed yet.")
    else:
        for item in st.session_state.history[:10]:
            badge = "🔴 FAKE" if item['verdict'] == 'FAKE' else "🟢 REAL"
            st.markdown(f"""
            <div class="history-item">
                <span style="font-size:11px;font-weight:700;">{badge}</span>
                <span style="font-size:11px;color:#6b6b6b;margin-left:0.5rem;">{item['confidence']}% confidence</span><br>
                <span style="font-size:12px;">{item['text']}</span>
            </div>""", unsafe_allow_html=True)

# ── SIDEBAR ──
with col_side:
    info = models.get('info', {})

    st.markdown('<div class="section-label">Session Statistics</div>', unsafe_allow_html=True)
    total = len(st.session_state.history)
    fake_count = sum(1 for h in st.session_state.history if h['verdict'] == 'FAKE')
    real_count = total - fake_count
    avg_conf = round(sum(h['confidence'] for h in st.session_state.history) / total, 1) if total else 0

    s1, s2 = st.columns(2)
    s1.metric("Analyzed", total)
    s2.metric("Avg Confidence", f"{avg_conf}%" if total else "—")
    s3, s4 = st.columns(2)
    s3.metric("🔴 Fake", fake_count)
    s4.metric("🟢 Real", real_count)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🗳 Scope: U.S. Politics Only</div>', unsafe_allow_html=True)
    st.markdown("""
    ✓ Elections & campaigns  
    ✓ Congress & legislation  
    ✓ White House & executive branch  
    ✓ U.S. political figures  
    ✓ Narrative political journalism  
    
    ✗ Sports, science, entertainment → unreliable
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🧠 Why Hybrid?</div>', unsafe_allow_html=True)
    st.markdown("""
    **TF-IDF** catches keyword patterns but flags narrative journalism as fake.
    
    **BERT** understands writing context and meaning beyond keyword frequency.
    
    **Meta-classifier** learns the optimal weight of each model's prediction.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    - Training samples: {info.get('training_samples', 44898):,}
    - Datasets: {info.get('datasets', 'Kaggle + LIAR + Synthetic')}
    - BERT: all-MiniLM-L6-v2
    - Hybrid accuracy: {round(info.get('hybrid_acc', 0)*100, 1)}%
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Known Limitations</div>', unsafe_allow_html=True)
    st.warning("""
    ! Detects writing style, not factual accuracy  
    ! Wire-style fake news may be missed  
    ! Narrative real journalism may be flagged  
    ! Always verify with trusted sources
    """)

    st.markdown("---")
    st.caption("Built by Saleh Salavudheen · UIUC CS · For educational use only")
