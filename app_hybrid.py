"""
app_hybrid.py — Flask backend for Hybrid TF-IDF + BERT Fake News Detector

Usage:
    python app_hybrid.py

Requires trained model files in model/ folder.
Run train_hybrid.py first.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import os
import re
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_DIR = 'model'

# Global model components
tfidf_model = None
bert_only_model = None
meta_model = None
vectorizer = None
scaler = None
bert_model = None
model_info = {}

def load_models():
    global tfidf_model, bert_only_model, meta_model, vectorizer, scaler, bert_model, model_info

    required = ['tfidf_model.pkl', 'bert_only_model.pkl', 'meta_model.pkl',
                'vectorizer.pkl', 'scaler.pkl']

    for f in required:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            print(f"ERROR: {f} not found. Run train_hybrid.py first.")
            return False

    print("Loading model components...")

    with open(os.path.join(MODEL_DIR, 'tfidf_model.pkl'), 'rb') as f:
        tfidf_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'bert_only_model.pkl'), 'rb') as f:
        bert_only_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'rb') as f:
        meta_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    info_path = os.path.join(MODEL_DIR, 'model_info.pkl')
    if os.path.exists(info_path):
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)

    print("Loading BERT model (this may take 30 seconds)...")
    bert_name = model_info.get('bert_model_name', 'all-MiniLM-L6-v2')
    bert_model = SentenceTransformer(bert_name)

    print("All models loaded successfully.")
    return True

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_hybrid(text):
    """Run full hybrid prediction pipeline"""

    # TF-IDF features (on cleaned text)
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    tfidf_proba = tfidf_model.predict_proba(vectorized)

    # BERT features (on original text — BERT handles punctuation/case)
    bert_text = text[:1500]
    bert_embedding = bert_model.encode([bert_text])
    bert_scaled = scaler.transform(bert_embedding)
    bert_proba = bert_only_model.predict_proba(bert_scaled)

    # Meta model combines both
    meta_features = np.hstack([tfidf_proba, bert_proba])
    final_proba = meta_model.predict_proba(meta_features)[0]
    final_pred = meta_model.predict(meta_features)[0]

    return {
        'prediction': int(final_pred),
        'fake_pct': round(float(final_proba[0]) * 100, 1),
        'real_pct': round(float(final_proba[1]) * 100, 1),
        'tfidf_fake_pct': round(float(tfidf_proba[0][0]) * 100, 1),
        'tfidf_real_pct': round(float(tfidf_proba[0][1]) * 100, 1),
        'bert_fake_pct': round(float(bert_proba[0][0]) * 100, 1),
        'bert_real_pct': round(float(bert_proba[0][1]) * 100, 1),
    }

def get_word_importance(text, n=6):
    """Get word importance from TF-IDF model (explainability component)"""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = tfidf_model.coef_[0]
    article_features = vectorized.toarray()[0]
    word_importance = article_features * coefficients

    present_indices = np.where(article_features > 0)[0]
    if len(present_indices) == 0:
        return [], []

    present_importance = word_importance[present_indices]
    present_words = feature_names[present_indices]
    sorted_idx = np.argsort(present_importance)

    fake_idx = sorted_idx[:n]
    fake_words = [
        {'word': present_words[i], 'score': round(abs(float(present_importance[i])) * 100, 1)}
        for i in fake_idx if present_importance[i] < 0
    ]
    fake_words = sorted(fake_words, key=lambda x: x['score'], reverse=True)

    real_idx = sorted_idx[-n:][::-1]
    real_words = [
        {'word': present_words[i], 'score': round(abs(float(present_importance[i])) * 100, 1)}
        for i in real_idx if present_importance[i] > 0
    ]
    real_words = sorted(real_words, key=lambda x: x['score'], reverse=True)

    return fake_words, real_words

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': tfidf_model is not None,
        'model_type': 'Hybrid TF-IDF + BERT',
        'bert_model': model_info.get('bert_model_name', 'unknown'),
        'hybrid_accuracy': round(model_info.get('hybrid_acc', 0) * 100, 2),
        'tfidf_accuracy': round(model_info.get('tfidf_acc', 0) * 100, 2),
        'bert_accuracy': round(model_info.get('bert_acc', 0) * 100, 2),
        'vocab_size': model_info.get('vocab_size', 0),
        'training_samples': model_info.get('training_samples', 0),
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if tfidf_model is None:
        return jsonify({'error': 'Models not loaded. Run train_hybrid.py first.'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text'].strip()
    if len(text) < 50:
        return jsonify({'error': 'Text too short. Please provide at least 50 characters.'}), 400

    result = predict_hybrid(text)
    fake_words, real_words = get_word_importance(text)

    verdict = 'FAKE' if result['prediction'] == 0 else 'REAL'
    confidence = result['fake_pct'] if verdict == 'FAKE' else result['real_pct']

    return jsonify({
        'verdict': verdict,
        'confidence': confidence,
        'fake_pct': result['fake_pct'],
        'real_pct': result['real_pct'],
        'tfidf_fake_pct': result['tfidf_fake_pct'],
        'tfidf_real_pct': result['tfidf_real_pct'],
        'bert_fake_pct': result['bert_fake_pct'],
        'bert_real_pct': result['bert_real_pct'],
        'fake_words': fake_words,
        'real_words': real_words,
        'char_count': len(text),
        'word_count': len(text.split()),
        'model_type': 'Hybrid TF-IDF + BERT',
    })

if __name__ == '__main__':
    if load_models():
        app.run(debug=False, port=5000)
    else:
        print("Failed to load models. Run train_hybrid.py first.")
