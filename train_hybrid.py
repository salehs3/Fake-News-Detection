"""
train_hybrid.py — Train and save the Hybrid TF-IDF + BERT Fake News Detection model
Run this ONCE before starting the server.

Usage:
    python train_hybrid.py

Requires:
    - Fake.csv and True.csv in a 'data/' folder
    - pip install pandas scikit-learn sentence-transformers torch numpy
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ── Config ──
DATA_DIR = 'data'
MODEL_DIR = 'model'
FAKE_CSV = os.path.join(DATA_DIR, 'Fake.csv')
TRUE_CSV = os.path.join(DATA_DIR, 'True.csv')
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'  # same model as Resume Screener

# How many articles to use for BERT embeddings
# BERT is slow — we sample for training but use full model at inference
# Increase this for better accuracy but much longer training time
BERT_SAMPLE_SIZE = 5000  # ~10-15 minutes on CPU

os.makedirs(MODEL_DIR, exist_ok=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_bert_embeddings(texts, model, batch_size=32):
    """Generate BERT embeddings in batches"""
    embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        print(f"   BERT: {min(i+batch_size, total)}/{total} articles processed", end='\r')
    print()
    return np.vstack(embeddings)

def main():
    print("=" * 60)
    print("TruthLens — Hybrid TF-IDF + BERT Model Training")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/7] Loading dataset...")
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        print(f"ERROR: Could not find dataset files.")
        print(f"Place Fake.csv and True.csv inside the '{DATA_DIR}/' folder.")
        return

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)
    fake_df['label'] = 0
    true_df['label'] = 1

    data = pd.concat([fake_df, true_df], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Loaded {len(data):,} articles ({len(fake_df):,} fake, {len(true_df):,} real)")

    # ── Preprocess ──
    print("\n[2/7] Preprocessing text...")
    data['content'] = data['title'].fillna('') + ' ' + data['text'].fillna('')
    data['content_clean'] = data['content'].apply(clean_text)

    # For BERT we use original text (not cleaned) — BERT understands punctuation and case
    data['content_bert'] = (data['title'].fillna('') + ' ' + data['text'].fillna('')).str.strip()
    # Truncate to 512 tokens worth of text for BERT
    data['content_bert'] = data['content_bert'].str[:1500]
    print(f"   Done.")

    # ── Split ──
    print("\n[3/7] Splitting train/test...")
    X_clean = data['content_clean']
    X_bert_text = data['content_bert']
    y = data['label']

    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

    # ── TF-IDF ──
    print("\n[4/7] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=50000)
    X_tfidf_train = vectorizer.fit_transform(X_clean.iloc[train_idx])
    X_tfidf_test = vectorizer.transform(X_clean.iloc[test_idx])
    print(f"   Vocabulary: {len(vectorizer.vocabulary_):,} terms")

    # Train TF-IDF only model first (for comparison)
    print("\n   Training TF-IDF baseline...")
    tfidf_model = LogisticRegression(max_iter=1000, random_state=42)
    tfidf_model.fit(X_tfidf_train, y.iloc[train_idx])
    tfidf_preds = tfidf_model.predict(X_tfidf_test)
    tfidf_acc = accuracy_score(y.iloc[test_idx], tfidf_preds)
    print(f"   TF-IDF only accuracy: {tfidf_acc*100:.2f}%")

    # ── BERT ──
    print(f"\n[5/7] Generating BERT embeddings (this takes time on CPU)...")
    print(f"   Loading {BERT_MODEL_NAME}...")
    bert_model = SentenceTransformer(BERT_MODEL_NAME)

    # Sample for training if dataset is large
    train_texts = X_bert_text.iloc[train_idx].tolist()
    test_texts = X_bert_text.iloc[test_idx].tolist()

    if len(train_texts) > BERT_SAMPLE_SIZE:
        print(f"   Sampling {BERT_SAMPLE_SIZE:,} articles for BERT training (full dataset too slow on CPU)")
        sample_idx = np.random.choice(len(train_texts), BERT_SAMPLE_SIZE, replace=False)
        train_texts_sample = [train_texts[i] for i in sample_idx]
        y_train_sample = y.iloc[train_idx].iloc[sample_idx]
        X_tfidf_train_sample = X_tfidf_train[sample_idx]
    else:
        train_texts_sample = train_texts
        y_train_sample = y.iloc[train_idx]
        X_tfidf_train_sample = X_tfidf_train

    print(f"   Encoding {len(train_texts_sample):,} training articles with BERT...")
    bert_train = get_bert_embeddings(train_texts_sample, bert_model)

    print(f"   Encoding {len(test_texts):,} test articles with BERT...")
    bert_test = get_bert_embeddings(test_texts, bert_model)

    # Scale BERT embeddings
    scaler = StandardScaler()
    bert_train_scaled = scaler.fit_transform(bert_train)
    bert_test_scaled = scaler.transform(bert_test)

    # ── BERT only model ──
    print("\n   Training BERT-only classifier...")
    bert_only_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    bert_only_model.fit(bert_train_scaled, y_train_sample)
    bert_preds = bert_only_model.predict(bert_test_scaled)
    bert_acc = accuracy_score(y.iloc[test_idx], bert_preds)
    print(f"   BERT only accuracy: {bert_acc*100:.2f}%")

    # ── Hybrid: combine TF-IDF probabilities + BERT probabilities ──
    print("\n[6/7] Building Hybrid model...")

    # Get probabilities from both models on test set
    tfidf_proba_test = tfidf_model.predict_proba(X_tfidf_test)
    bert_proba_test = bert_only_model.predict_proba(bert_test_scaled)

    # Get probabilities on training set for meta-model
    tfidf_proba_train = tfidf_model.predict_proba(X_tfidf_train_sample)
    bert_proba_train = bert_only_model.predict_proba(bert_train_scaled)

    # Stack probabilities as features for meta-classifier
    X_meta_train = np.hstack([tfidf_proba_train, bert_proba_train])
    X_meta_test = np.hstack([tfidf_proba_test, bert_proba_test])

    # Train meta-classifier (learns optimal weighting of TF-IDF vs BERT)
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_meta_train, y_train_sample)

    hybrid_preds = meta_model.predict(X_meta_test)
    hybrid_acc = accuracy_score(y.iloc[test_idx], hybrid_preds)

    print(f"\n{'='*60}")
    print(f"   TF-IDF only:  {tfidf_acc*100:.2f}%")
    print(f"   BERT only:    {bert_acc*100:.2f}%")
    print(f"   Hybrid:       {hybrid_acc*100:.2f}%  ← selected")
    print(f"{'='*60}")
    print("\n" + classification_report(y.iloc[test_idx], hybrid_preds, target_names=['Fake', 'Real']))

    # ── Save everything ──
    print("[7/7] Saving all model components...")

    with open(os.path.join(MODEL_DIR, 'tfidf_model.pkl'), 'wb') as f:
        pickle.dump(tfidf_model, f)
    with open(os.path.join(MODEL_DIR, 'bert_only_model.pkl'), 'wb') as f:
        pickle.dump(bert_only_model, f)
    with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'wb') as f:
        pickle.dump(meta_model, f)
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save model info
    model_info = {
        'tfidf_acc': tfidf_acc,
        'bert_acc': bert_acc,
        'hybrid_acc': hybrid_acc,
        'bert_model_name': BERT_MODEL_NAME,
        'vocab_size': len(vectorizer.vocabulary_),
        'training_samples': len(data),
    }
    with open(os.path.join(MODEL_DIR, 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)

    print(f"\n✅ All model components saved to {MODEL_DIR}/")
    print("   Run: python app_hybrid.py   to start the server")
    print("=" * 60)

if __name__ == '__main__':
    main()
