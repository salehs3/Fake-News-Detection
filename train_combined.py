"""
train_combined.py — Train Hybrid model on Kaggle + LIAR datasets
for more robust fake news detection including narrative journalism.

Usage:
    python train_combined.py

Dataset setup:
    data/
    ├── Fake.csv          ← Kaggle dataset
    ├── True.csv          ← Kaggle dataset
    ├── train.tsv         ← LIAR dataset (download below)
    ├── test.tsv          ← LIAR dataset
    └── valid.tsv         ← LIAR dataset

Download LIAR:
    https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    Extract the 3 .tsv files into your data/ folder
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
from sklearn.utils import resample
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
MODEL_DIR = 'model'
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'
BERT_SAMPLE_SIZE = 6000

os.makedirs(MODEL_DIR, exist_ok=True)

# ── LIAR label mapping ──
# Convert 6-class LIAR labels to binary fake/real
# We drop ambiguous middle labels (half-true, barely-true, mostly-false)
LIAR_FAKE_LABELS = {'pants-fire', 'false'}
LIAR_REAL_LABELS = {'true', 'mostly-true'}
LIAR_SKIP_LABELS = {'half-true', 'barely-true', 'mostly-false'}

# ── CURATED SYNTHETIC DOCUMENTS ──
# These are hand-crafted examples of real journalism and fake news
# Each document is repeated multiple times to give it training weight
# This directly teaches the model what credible vs sensational writing looks like

REAL_JOURNALISM_DOCS = [
    # Wire service / formal political reporting style
    "according to official statement the senator confirmed the legislation passed committee vote bipartisan support",
    "the federal court ruled the government attorney filed motion judge approved hearing scheduled next month",
    "the white house press secretary confirmed the president signed executive order following congressional approval",
    "reuters reported the department of justice investigation concluded findings submitted to federal prosecutors",
    "the official statement released by the state department confirmed diplomatic negotiations ongoing between governments",
    "according to court documents the attorney general filed charges following grand jury indictment proceedings",
    "the congressional budget office released analysis projecting deficit reduction over next decade according to report",
    "associated press confirmed election results certified by state officials following recount procedures completed",
    "the supreme court issued ruling majority opinion written by justice citing constitutional precedent established",
    "committee hearing testimony provided under oath witnesses confirmed details documented in official report",

    # Narrative investigative journalism style (FT, New Yorker, longform)
    "in the nineteen seventies the ambitious young developer sought counsel from the notorious new york lawyer",
    "the relationship between the two men would have profound consequences for american politics and governance",
    "the brash attorney whose clientele included figures from various walks of life offered candid strategic advice",
    "over the course of their long association the developer absorbed lessons about aggressive litigation tactics",
    "the federal lawsuit alleged discrimination in housing policies the defendants contested the charges vigorously",
    "historical records documents and court filings reveal the extent of the legal battle that followed",
    "the attorney told his client to fight the case in court and let prosecutors prove their allegations",
    "investigative journalism requires examining court records interviews with sources and documentary evidence",
    "the biographer spent years researching public records interviewing associates and reviewing correspondence",
    "the story of how american political culture shifted can be traced through the careers of key figures",

    # Academic and policy reporting
    "researchers at the university published findings suggesting correlation between economic conditions and voting patterns",
    "the policy institute analysis concluded regulatory changes would affect market competition according to economists",
    "data from the census bureau indicated demographic shifts across multiple metropolitan areas over past decade",
    "the nonpartisan congressional research service issued report examining implications of proposed legislation",
    "economists from both political parties agreed the fiscal impact required further analysis before implementation",

    # Credible political reporting with names and places
    "the senator from new york appeared before the judiciary committee answering questions about campaign finance",
    "the governor announced infrastructure investment plan citing federal funding allocation approved by congress",
    "former secretary of state testified before house committee providing detailed account of diplomatic communications",
    "the mayor confirmed city council approved budget following months of negotiations between competing priorities",
    "political analysts noted the election results reflected shifting demographics in suburban congressional districts",

    # Casual language used in REAL credible journalism
    # These teach the model that casual words like "just", "like", "let" appear in real news too
    "the senator just confirmed the vote will happen next week according to spokesperson",
    "officials said they would just let the courts decide the matter without further comment",
    "the attorney told his client to just fight the case and let the evidence speak",
    "analysts said the bill is just the first step in a much longer legislative process",
    "the governor said he would just sign whatever the legislature sent to his desk",
    "sources familiar with the matter said the talks are just getting started",
    "the judge ruled they should just proceed with the trial as scheduled next month",
    "the spokesperson said the department would just follow existing guidelines on the matter",
    "economists said the data is just one indicator among many that should be considered",
    "the committee said they would just wait for the full report before taking action",

    # "like" used in real journalism context
    "organizations like the aclu and naacp filed briefs supporting the constitutional challenge",
    "institutions like the federal reserve and treasury coordinate monetary policy decisions",
    "figures like the senator and governor have shaped american political discourse for decades",
    "cities like new york and chicago have seen significant shifts in voting patterns recently",
    "countries like canada and germany have adopted similar approaches to healthcare legislation",
    "publications like the financial times and wall street journal covered the regulatory changes",
    "agencies like the fbi and department of justice coordinated on the investigation together",
    "programs like medicare and social security remain central to domestic policy debates",
    "leaders like the president and secretary of state met to discuss diplomatic relations",
    "committees like the judiciary and finance panels have oversight authority over the matter",

    # Trump mentioned in credible real journalism context
    "trump signed the legislation into law following senate approval of the bipartisan bill",
    "the trump administration confirmed the policy change in an official white house statement",
    "trump appeared before reporters tuesday confirming the decision had been made official",
    "court documents filed by trump attorneys cited constitutional precedent in their arguments",
    "trump met with congressional leaders wednesday to discuss infrastructure funding priorities",
    "the former president trump confirmed his position in a written statement to the press",
    "trump administration officials told reporters the investigation had concluded its findings",

    # "black" in real journalism context
    "black voters in key swing states shifted their preferences according to exit poll data",
    "black lawmakers on the committee questioned the administration official during testimony",
    "the supreme court ruling on black voter representation drew reactions from both parties",
    "black community leaders met with the mayor to discuss policing and education policy",
    "black americans represent a significant voting bloc that both parties have courted actively",

    # Historical and investigative journalism with casual language
    "looking back at the events it becomes clear just how significant the decision really was",
    "the story is really just about how power operates in american political institutions",
    "what makes this case interesting is just how ordinary the participants seemed at the time",
    "historians now see the period as just the beginning of a much larger transformation",
    "the records show just how deeply the relationship shaped subsequent political developments",
]

FAKE_NEWS_DOCS = [
    # Sensational breaking news style
    "breaking bombshell revelation exposes shocking secret globalist agenda patriots must share before deleted",
    "explosive report deep state operatives caught red handed betraying america mainstream media refuses cover",
    "breaking news whistleblower exposes radical left conspiracy that will shock every true american patriot",
    "urgent alert deep state trying silence this bombshell truth they desperately do not want you knowing",
    "shocking revelation exposes globalist plot mainstream media blackout you need see this immediately share",

    # Extreme partisan language
    "radical leftist socialist agenda destroying america real patriots standing against corrupt deep state swamp",
    "the radical left wing extremists pushing socialist agenda mainstream media covering up truth from americans",
    "corrupt politicians rigging system against hardworking american people real truth being suppressed exposed",
    "mainstream media fake news propaganda machine covering up truth patriots fighting back against deep state",
    "liberal elites globalists destroying traditional american values patriots fighting back exposing corrupt agenda",

    # Conspiracy and misinformation patterns
    "government covering up truth about secret agenda share this before they delete it wake up america",
    "they do not want you to know this bombshell truth being suppressed by corrupt establishment elites",
    "insider sources reveal shocking conspiracy that mainstream media refuses to report share with everyone now",
    "banned from social media this explosive truth exposes globalist agenda threatening american freedom liberty",
    "the truth they are hiding from you shocking revelation about secret plot against american people exposed",

    # Clickbait and emotional manipulation
    "you will not believe what they just did to betray america this is absolutely outrageous unbelievable",
    "this is the most shocking thing ever done in american history mainstream media completely silent why",
    "real americans are furious about this outrageous betrayal of our great nation share if you agree",
    "this single fact destroys the entire liberal narrative and they cannot handle the truth being exposed",
    "incredible what they are getting away with while mainstream media looks the other way share this now",
]

# How many times to repeat each synthetic document during training
# Higher = more influence on the model
REAL_DOC_REPEAT = 25
FAKE_DOC_REPEAT = 25

def build_synthetic_data():
    """Build synthetic training documents from curated word lists"""
    real_docs = REAL_JOURNALISM_DOCS * REAL_DOC_REPEAT
    fake_docs = FAKE_NEWS_DOCS * FAKE_DOC_REPEAT

    synthetic = pd.DataFrame({
        'content': real_docs + fake_docs,
        'label': [1] * len(real_docs) + [0] * len(fake_docs),
        'source': ['synthetic_real'] * len(real_docs) + ['synthetic_fake'] * len(fake_docs)
    })
    print(f"   Synthetic: {len(real_docs):,} real + {len(fake_docs):,} fake = {len(synthetic):,} total")
    return synthetic

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    # Remove short words (1-2 chars) which are usually artifacts
    # e.g. "st" from "1st", "nd" from "2nd" after number stripping
    text = ' '.join([w for w in text.split() if len(w) > 2])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_kaggle_data():
    """Load original Kaggle fake/real news dataset"""
    fake_path = os.path.join(DATA_DIR, 'Fake.csv')
    true_path = os.path.join(DATA_DIR, 'True.csv')

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("ERROR: Fake.csv or True.csv not found in data/")
        return None

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 0
    true_df['label'] = 1

    data = pd.concat([fake_df, true_df], ignore_index=True)
    data['content'] = data['title'].fillna('') + ' ' + data['text'].fillna('')
    data['source'] = 'kaggle'

    print(f"   Kaggle: {len(fake_df):,} fake + {len(true_df):,} real = {len(data):,} total")
    return data[['content', 'label', 'source']]

def load_liar_data():
    """Load LIAR dataset from TSV files"""
    liar_files = ['train.tsv', 'test.tsv', 'valid.tsv']
    dfs = []

    for fname in liar_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"   WARNING: {fname} not found — skipping")
            continue

        # LIAR TSV columns:
        # 0:id, 1:label, 2:statement, 3:subject, 4:speaker,
        # 5:job, 6:state, 7:party, 8-12:credit history
        df = pd.read_csv(fpath, sep='\t', header=None,
                        names=['id','label','statement','subject','speaker',
                               'job','state','party','barely_true','false_count',
                               'half_true','mostly_true','pants_fire'])
        dfs.append(df)

    if not dfs:
        return None

    liar = pd.concat(dfs, ignore_index=True)

    # Filter to clear fake/real only
    liar_clear = liar[liar['label'].isin(LIAR_FAKE_LABELS | LIAR_REAL_LABELS)].copy()

    # Binary labels
    liar_clear['label_binary'] = liar_clear['label'].apply(
        lambda x: 0 if x in LIAR_FAKE_LABELS else 1
    )

    # Enrich text: statement + speaker context for more signal
    liar_clear['content'] = (
        liar_clear['statement'].fillna('') + ' ' +
        liar_clear['speaker'].fillna('') + ' ' +
        liar_clear['subject'].fillna('')
    )

    fake_count = (liar_clear['label_binary'] == 0).sum()
    real_count = (liar_clear['label_binary'] == 1).sum()
    print(f"   LIAR: {fake_count:,} fake + {real_count:,} real = {len(liar_clear):,} usable")
    print(f"   (Skipped {len(liar)-len(liar_clear):,} ambiguous labels: half-true, barely-true, mostly-false)")

    return liar_clear[['content', 'label_binary', 'label']].rename(
        columns={'label_binary': 'label', 'label': 'liar_label'}
    ).assign(source='liar')

def get_bert_embeddings(texts, model, batch_size=32):
    embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_emb)
        print(f"   BERT: {min(i+batch_size, total)}/{total} processed", end='\r')
    print()
    return np.vstack(embeddings)

def get_word_importance_export(vectorizer, tfidf_model, n=20):
    """Export top fake/real words for reference"""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = tfidf_model.coef_[0]
    sorted_idx = np.argsort(coefficients)

    fake_words = [(feature_names[i], round(float(coefficients[i]), 4))
                  for i in sorted_idx[:n]]
    real_words = [(feature_names[i], round(float(coefficients[i]), 4))
                  for i in sorted_idx[-n:][::-1]]
    return fake_words, real_words

def main():
    print("=" * 65)
    print("TruthLens — Combined Kaggle + LIAR Hybrid Model Training")
    print("=" * 65)

    # ── Load datasets ──
    print("\n[1/8] Loading datasets...")
    kaggle_data = load_kaggle_data()
    liar_data = load_liar_data()

    if kaggle_data is None:
        print("Cannot proceed without Kaggle data.")
        return

    # ── Combine ──
    print("\n[2/8] Combining datasets...")
    print("   Building synthetic curated word list documents...")
    synthetic_data = build_synthetic_data()

    if liar_data is not None:
        liar_weighted = pd.concat([liar_data] * 2, ignore_index=True)
        combined = pd.concat([
            kaggle_data[['content', 'label', 'source']],
            liar_weighted[['content', 'label', 'source']],
            synthetic_data[['content', 'label', 'source']]
        ], ignore_index=True)
        print(f"   Kaggle: {len(kaggle_data):,} · LIAR: {len(liar_weighted):,} · Synthetic: {len(synthetic_data):,}")
    else:
        combined = pd.concat([
            kaggle_data[['content', 'label', 'source']],
            synthetic_data[['content', 'label', 'source']]
        ], ignore_index=True)
        print("   LIAR not found — using Kaggle + Synthetic only")
        print(f"   Kaggle: {len(kaggle_data):,} · Synthetic: {len(synthetic_data):,}")
    print(f"   Total combined: {len(combined):,} samples")

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop any rows with missing labels or content, force label to int
    combined = combined.dropna(subset=['label', 'content'])
    combined['label'] = combined['label'].astype(int)

    fake_total = (combined['label'] == 0).sum()
    real_total = (combined['label'] == 1).sum()
    print(f"   Class balance: {fake_total:,} fake ({fake_total/len(combined)*100:.1f}%) · {real_total:,} real ({real_total/len(combined)*100:.1f}%)")

    # ── Preprocess ──
    print("\n[3/8] Preprocessing text...")
    combined['content_clean'] = combined['content'].apply(clean_text)
    combined['content_bert'] = combined['content'].str.strip().str[:1500]
    print(f"   Done.")

    # ── Split ──
    print("\n[4/8] Train/test split...")
    indices = np.arange(len(combined))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=combined['label']
    )
    print(f"   Train: {len(train_idx):,} · Test: {len(test_idx):,}")

    # ── TF-IDF ──
    print("\n[5/8] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=60000)
    X_tfidf_train = vectorizer.fit_transform(combined['content_clean'].iloc[train_idx])
    X_tfidf_test = vectorizer.transform(combined['content_clean'].iloc[test_idx])
    y_train = combined['label'].iloc[train_idx]
    y_test = combined['label'].iloc[test_idx]
    print(f"   Vocabulary: {len(vectorizer.vocabulary_):,} terms")

    print("\n   Training TF-IDF model...")
    tfidf_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    tfidf_model.fit(X_tfidf_train, y_train)
    tfidf_acc = accuracy_score(y_test, tfidf_model.predict(X_tfidf_test))
    print(f"   TF-IDF accuracy: {tfidf_acc*100:.2f}%")

    # Top words for reference
    fake_words, real_words = get_word_importance_export(vectorizer, tfidf_model)
    print(f"   Top fake indicators: {[w for w,s in fake_words[:5]]}")
    print(f"   Top real indicators: {[w for w,s in real_words[:5]]}")

    # ── BERT ──
    print(f"\n[6/8] Generating BERT embeddings (slow on CPU — ~20 min)...")
    print(f"   Loading {BERT_MODEL_NAME}...")
    bert_model = SentenceTransformer(BERT_MODEL_NAME)

    train_texts = combined['content_bert'].iloc[train_idx].tolist()
    test_texts = combined['content_bert'].iloc[test_idx].tolist()

    if len(train_texts) > BERT_SAMPLE_SIZE:
        print(f"   Sampling {BERT_SAMPLE_SIZE:,} for BERT training...")
        sample_idx = np.random.RandomState(42).choice(
            len(train_texts), BERT_SAMPLE_SIZE, replace=False
        )
        train_texts_sample = [train_texts[i] for i in sample_idx]
        y_train_sample = y_train.iloc[sample_idx]
        X_tfidf_train_sample = X_tfidf_train[sample_idx]
    else:
        train_texts_sample = train_texts
        y_train_sample = y_train
        X_tfidf_train_sample = X_tfidf_train

    print(f"   Encoding {len(train_texts_sample):,} training texts...")
    bert_train = get_bert_embeddings(train_texts_sample, bert_model)
    print(f"   Encoding {len(test_texts):,} test texts...")
    bert_test = get_bert_embeddings(test_texts, bert_model)

    scaler = StandardScaler()
    bert_train_scaled = scaler.fit_transform(bert_train)
    bert_test_scaled = scaler.transform(bert_test)

    print("\n   Training BERT-only classifier...")
    bert_only_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    bert_only_model.fit(bert_train_scaled, y_train_sample)
    bert_acc = accuracy_score(y_test, bert_only_model.predict(bert_test_scaled))
    print(f"   BERT accuracy: {bert_acc*100:.2f}%")

    # ── Hybrid meta-model ──
    print("\n[7/8] Training hybrid meta-model...")
    tfidf_proba_train = tfidf_model.predict_proba(X_tfidf_train_sample)
    bert_proba_train = bert_only_model.predict_proba(bert_train_scaled)
    tfidf_proba_test = tfidf_model.predict_proba(X_tfidf_test)
    bert_proba_test = bert_only_model.predict_proba(bert_test_scaled)

    X_meta_train = np.hstack([tfidf_proba_train, bert_proba_train])
    X_meta_test = np.hstack([tfidf_proba_test, bert_proba_test])

    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_meta_train, y_train_sample)

    hybrid_preds = meta_model.predict(X_meta_test)
    hybrid_acc = accuracy_score(y_test, hybrid_preds)

    print(f"\n{'='*65}")
    print(f"   TF-IDF only:  {tfidf_acc*100:.2f}%")
    print(f"   BERT only:    {bert_acc*100:.2f}%")
    print(f"   Hybrid:       {hybrid_acc*100:.2f}%  ← selected")
    print(f"{'='*65}")
    print("\n" + classification_report(y_test, hybrid_preds, target_names=['Fake', 'Real']))

    # ── Save ──
    print("[8/8] Saving model components...")
    components = {
        'tfidf_model.pkl': tfidf_model,
        'bert_only_model.pkl': bert_only_model,
        'meta_model.pkl': meta_model,
        'vectorizer.pkl': vectorizer,
        'scaler.pkl': scaler,
    }
    for fname, obj in components.items():
        with open(os.path.join(MODEL_DIR, fname), 'wb') as f:
            pickle.dump(obj, f)

    model_info = {
        'tfidf_acc': tfidf_acc,
        'bert_acc': bert_acc,
        'hybrid_acc': hybrid_acc,
        'bert_model_name': BERT_MODEL_NAME,
        'vocab_size': len(vectorizer.vocabulary_),
        'training_samples': len(combined),
        'kaggle_samples': len(kaggle_data),
        'liar_samples': len(liar_data) if liar_data is not None else 0,
        'datasets': 'Kaggle + LIAR' if liar_data is not None else 'Kaggle only',
    }
    with open(os.path.join(MODEL_DIR, 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)

    print(f"\n✅ All components saved to {MODEL_DIR}/")
    print(f"   Training data: {model_info['datasets']}")
    print(f"   Run: python app_hybrid.py   to start the server")
    print("=" * 65)

if __name__ == '__main__':
    main()
