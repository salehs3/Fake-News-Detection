# Fake News Detection using NLP and Machine Learning

## Overview
This project builds a machine learning pipeline to classify 
news articles as fake or real using Natural Language Processing 
techniques. The project goes beyond basic classification by 
including explainability analysis, error analysis, and critical 
evaluation of model limitations.

## Dataset
Fake and Real News Dataset from Kaggle containing 44,898 articles:
- 23,481 fake news articles (label = 0)
- 21,417 real news articles (label = 1)

## Technologies
- Python
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- NumPy

## Pipeline
1. Data loading and labeling
2. Text preprocessing (lowercasing, removing punctuation)
3. Feature extraction using TF-IDF vectorization
4. Train/test split (80/20)
5. Model training and comparison
6. Explainability analysis
7. Error analysis

## Models Compared
| Model | Accuracy | F1 Score | Training Speed |
|-------|----------|----------|----------------|
| Logistic Regression | 99% | 0.99 | Fast |
| Naive Bayes | 94% | 0.94 | Very Fast |
| Random Forest | 99% | 0.99 | Very Slow |

Logistic Regression was selected as the best model — it achieved 
the same accuracy as Random Forest at a fraction of the 
computational cost.

## Explainability
The project includes word level explainability showing which 
specific words drove each prediction toward fake or real. 
This was implemented using logistic regression coefficients 
combined with TF-IDF scores to calculate per word importance 
for individual predictions.

## Error Analysis
The model achieved 98.79% accuracy with 109 errors out of 
8980 test articles.

Key findings:
- Errors were evenly split: 55 false negatives, 54 false positives
- Political articles accounted for the majority of errors 
  (59 out of 109)
- Misclassified articles were on average 390 characters longer 
  than correctly classified ones
- Text normalization may have removed useful signals like ALL CAPS 
  writing style characteristic of sensational fake news headlines

## Limitations
- The high accuracy (99%) likely reflects stylistic differences 
  between dataset sources rather than genuine semantic understanding 
  of misinformation
- The model was tested only on one dataset — performance on 
  real world unseen data would likely be significantly lower
- TF-IDF does not capture word order or context — the word 
  "not guilty" would be treated the same as "guilty"
- Political articles are significantly harder to classify 
  due to overlapping writing styles between fake and real news

## What I Would Improve
- Test on an out of domain dataset to measure real world performance
- Replace TF-IDF with BERT embeddings to capture context and 
  word relationships
- Preserve capitalization as a feature since ALL CAPS is a 
  strong signal for fake news headlines
- Collect more diverse training data covering different political 
  viewpoints and writing styles

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset from Kaggle: Fake and Real News Dataset
4. Place Fake.csv and True.csv in the data/ folder
5. Run notebooks/fake_news_analysis.ipynb

## Project Structure
fake-news-detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/
│   └── fake_news_analysis.ipynb
│
├── src/
│   └── predict.py
│
├── README.md
└── requirements.txt