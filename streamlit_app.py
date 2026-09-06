import streamlit as st
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="TruthLens — News Credibility Checker",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp, .stApp > div, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 780px !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── MASTHEAD ── */
    .masthead {
        text-align: center;
        padding: 1rem 0 1rem;
        border-bottom: 3px solid #1a1a1a;
        margin-bottom: 1.2rem;
    }
    .masthead h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        color: #1a1a1a;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .masthead h1 span { color: #c0392b; }
    .masthead-sub {
        font-size: 13px;
        color: #6b6b6b;
        margin-top: 0.4rem;
    }
    .scope-pill {
        display: inline-block;
        background: #3C3B6E;
        color: white;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-top: 0.6rem;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e8e2d9;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #888;
        font-size: 13px;
        font-weight: 600;
        padding: 8px 20px;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #c0392b !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── BUTTON ── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.03em !important;
    }

    /* ── VERDICT ── */
    .verdict-fake {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
        border: 2px solid #c0392b;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 0.8rem 0;
    }
    .verdict-real {
        background: linear-gradient(135deg, #f0fff4 0%, #e6ffed 100%);
        border: 2px solid #1a6b3c;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 0.8rem 0;
    }
    .verdict-emoji { font-size: 2.8rem; line-height: 1; margin-bottom: 0.5rem; }
    .verdict-title-fake {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #c0392b;
        margin: 0;
    }
    .verdict-title-real {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1a6b3c;
        margin: 0;
    }
    .verdict-sub { font-size: 13px; color: #6b6b6b; margin-top: 0.4rem; }

    /* ── CONFIDENCE BAR ── */
    .conf-wrap { margin: 0.8rem 0 0.4rem; }
    .conf-label {
        font-size: 11px;
        font-weight: 700;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .conf-track {
        background: #f0ece6;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .conf-fill-fake {
        height: 100%;
        background: linear-gradient(90deg, #922b21, #c0392b);
        border-radius: 6px;
    }
    .conf-fill-real {
        height: 100%;
        background: linear-gradient(90deg, #1a6b3c, #27ae60);
        border-radius: 6px;
    }
    .conf-value {
        font-size: 13px;
        font-weight: 700;
        margin-top: 0.3rem;
        text-align: right;
    }
    .conf-value.fake { color: #c0392b; }
    .conf-value.real { color: #1a6b3c; }

    /* ── SIGNAL BOX ── */
    .signal-box {
        background: #fafaf8;
        border: 1px solid #e8e2d9;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .signal-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #999;
        margin-bottom: 0.8rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #e8e2d9;
    }
    .signal-col-label {
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .signal-col-label.fake { color: #c0392b; }
    .signal-col-label.real { color: #1a6b3c; }
    .pill-fake {
        display: inline-block;
        background: #fdecea;
        color: #c0392b;
        border: 1px solid #f5b7b1;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 500;
        margin: 3px 3px 3px 0;
    }
    .pill-real {
        display: inline-block;
        background: #e8f5ee;
        color: #1a6b3c;
        border: 1px solid #a9dfbf;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 500;
        margin: 3px 3px 3px 0;
    }

    /* ── STEP ROWS ── */
    .step-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 0.8rem;
        align-items: flex-start;
    }
    .step-num {
        background: #1a1a1a;
        color: white;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        font-size: 11px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .step-text { font-size: 13px; color: #555; line-height: 1.5; }
    .step-text strong { color: #1a1a1a; }

    /* ── HISTORY ── */
    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #f0ece6;
        gap: 1rem;
    }
    .history-text {
        font-size: 13px;
        color: #666;
        flex: 1;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .badge-fake {
        background: #fdecea;
        color: #c0392b;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 10px;
        white-space: nowrap;
    }
    .badge-real {
        background: #e8f5ee;
        color: #1a6b3c;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 10px;
        white-space: nowrap;
    }

    .disclaimer {
        font-size: 11px;
        color: #aaa;
        text-align: center;
        margin-top: 0.4rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #e8e2d9;
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = 'model'

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
    }

def get_word_importance(text, models, n=8):
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
    fake_words = [present_words[i] for i in sorted_idx[:n] if present_importance[i] < 0]
    real_words = [present_words[i] for i in sorted_idx[-n:][::-1] if present_importance[i] > 0]
    return fake_words, real_words

if 'history' not in st.session_state:
    st.session_state.history = []

# ── MASTHEAD ──
st.markdown("""
<div class="masthead">
    <h1>Truth<span>Lens</span></h1>
    <div class="masthead-sub">Paste a news article and find out if it looks credible</div>
    <div><span class="scope-pill">🇺🇸 Best for U.S. Political News</span></div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models..."):
    models, error = load_all_models()

if error:
    st.error(f"Could not load models: {error}")
    st.stop()

# ── TABS ──
tab1, tab2 = st.tabs(["📋 Paste Article", "🔗 Check a URL"])
article_text = ""

with tab1:
    pasted_text = st.text_area(
        "article",
        height=180,
        placeholder="Paste the full text of a political news article here...",
        key="pasted_text",
        label_visibility="collapsed"
    )
    if pasted_text:
        wc = len(pasted_text.split())
        st.caption(f"{wc} words {'✅' if wc > 50 else '— try a longer article for better results'}")
    article_text = pasted_text

with tab2:
    url_input = st.text_input(
        "url",
        placeholder="https://www.reuters.com/article/...",
        label_visibility="collapsed"
    )
    fetch_btn = st.button("📥 Fetch Article", use_container_width=True)
    if fetch_btn and url_input:
        try:
            from newspaper import Article
            with st.spinner("Fetching..."):
                a = Article(url_input)
                a.download()
                a.parse()
                fetched = a.text.strip()
            if len(fetched) < 50:
                st.error("Couldn't extract text. Try pasting manually.")
            else:
                st.session_state.fetched_text = fetched
                st.success(f"✅ Fetched — {len(fetched.split())} words{' · ' + a.title if a.title else ''}")
        except ImportError:
            st.error("Run: pip install newspaper3k lxml_html_clean")
        except Exception:
            st.error("Site blocked the request. Try pasting the text manually.")
    if 'fetched_text' in st.session_state and st.session_state.fetched_text:
        article_text = st.session_state.fetched_text
        with st.expander("📄 View fetched text"):
            st.text(st.session_state.fetched_text[:800] + "...")

analyze_btn = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
st.markdown('<p class="disclaimer">Checks writing style patterns — not whether facts are true. Always verify with trusted sources.</p>', unsafe_allow_html=True)

# ── RESULT ──
if analyze_btn:
    if len(article_text.strip()) < 50:
        st.error("Please paste at least a few sentences for a reliable result.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(article_text, models)
            fake_words, real_words = get_word_importance(article_text, models)

        verdict = 'FAKE' if result['prediction'] == 0 else 'REAL'
        confidence = result['fake_pct'] if verdict == 'FAKE' else result['real_pct']
        is_fake = verdict == 'FAKE'

        if is_fake:
            st.markdown(f"""
            <div class="verdict-fake">
                <div class="verdict-emoji">🚨</div>
                <div class="verdict-title-fake">Likely Fake News</div>
                <div class="verdict-sub">The AI is {confidence}% confident this article shows signs of misinformation</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-real">
                <div class="verdict-emoji">✅</div>
                <div class="verdict-title-real">Looks Credible</div>
                <div class="verdict-sub">The AI is {confidence}% confident this article appears to be legitimate news</div>
            </div>""", unsafe_allow_html=True)

        fill = "fake" if is_fake else "real"
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-label">{'Suspicion' if is_fake else 'Credibility'} Score</div>
            <div class="conf-track">
                <div class="conf-fill-{fill}" style="width:{confidence}%"></div>
            </div>
            <div class="conf-value {fill}">{confidence}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="signal-box">', unsafe_allow_html=True)
        st.markdown('<div class="signal-header">🔎 Words that influenced this result</div>', unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown('<div class="signal-col-label fake">🚩 Raised concern</div>', unsafe_allow_html=True)
            if fake_words:
                st.markdown(' '.join([f'<span class="pill-fake">{w}</span>' for w in fake_words[:6]]), unsafe_allow_html=True)
            else:
                st.caption("None found")
        with wc2:
            st.markdown('<div class="signal-col-label real">✓ Looked credible</div>', unsafe_allow_html=True)
            if real_words:
                st.markdown(' '.join([f'<span class="pill-real">{w}</span>' for w in real_words[:6]]), unsafe_allow_html=True)
            else:
                st.caption("None found")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("⚠️ What this tool can and can't do"):
            st.markdown("""
**Can do:** Detect sensational writing patterns common in fake news.

**Cannot do:** Verify facts, catch sophisticated fakes written in professional style, or replace your own judgment.

**Verify further:** [Snopes](https://www.snopes.com) · [PolitiFact](https://www.politifact.com) · [FactCheck.org](https://www.factcheck.org)
            """)

        st.session_state.history.insert(0, {
            'text': article_text[:80] + '...',
            'verdict': verdict,
            'confidence': confidence
        })

st.markdown('<hr class="divider">', unsafe_allow_html=True)

with st.expander("🧠 How does TruthLens work?"):
    st.markdown("""
    <div>
        <div class="step-row"><div class="step-num">1</div><div class="step-text"><strong>You paste an article.</strong> TruthLens reads the full text.</div></div>
        <div class="step-row"><div class="step-num">2</div><div class="step-text"><strong>It scans word patterns.</strong> Fake news often uses emotional or vague language. Real news uses specific names, dates, and formal language.</div></div>
        <div class="step-row"><div class="step-num">3</div><div class="step-text"><strong>A second AI reads for meaning</strong> — understanding context, not just keywords.</div></div>
        <div class="step-row"><div class="step-num">4</div><div class="step-text"><strong>Both signals combine</strong> into a final credibility score.</div></div>
        <div class="step-row"><div class="step-num">5</div><div class="step-text"><strong>Trained on 44,000 articles</strong> of real and fake political news.</div></div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.history:
    with st.expander(f"📋 Recent checks ({len(st.session_state.history)})"):
        for item in st.session_state.history[:10]:
            badge = f'<span class="badge-fake">FAKE {item["confidence"]}%</span>' if item['verdict'] == 'FAKE' \
                    else f'<span class="badge-real">REAL {item["confidence"]}%</span>'
            st.markdown(f'<div class="history-item"><span class="history-text">{item["text"]}</span>{badge}</div>', unsafe_allow_html=True)

st.caption("TruthLens · UIUC CS · Educational use only · Not a substitute for critical thinking")
