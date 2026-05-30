# NLP Projects — AIUB Group 7

Two NLP projects built for the Natural Language Processing course at American International University-Bangladesh (AIUB), Fall 2025-2026.

---

## Project 1: Fake News Detection (Mid-Term)

Detects fake vs real news articles using classical NLP techniques and Naive Bayes classification.

**Tech:** Python, NLTK, scikit-learn, pandas

**Pipeline:** Tokenization → Lowercase → Synonym Substitution → Stemming → Lemmatization → Punctuation Removal → Stopword Removal → TF-IDF → Naive Bayes

**Results:**
| Metric | Score |
|--------|-------|
| Accuracy | 96.5% |
| Precision | 96.5% |
| Recall | 96.4% |
| F1-Score | 96.4% |

---

## Project 2: AI-Generated Content Detection (Final)

Detects whether text was written by a human or AI by fine-tuning BERT and RoBERTa on 5,000 essays.

**Dataset:** [AI vs Human Text — Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data)

**Tech:** Python, PyTorch, HuggingFace Transformers, scikit-learn

**Results:**
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|----|---------|
| BERT | 0.9773 | 0.9717 | 0.9683 | 0.9700 | 0.997 |
| RoBERTa | 0.9680 | 0.9248 | 0.9965 | 0.9593 | 0.9996 |

**Key finding:** BERT wins on accuracy and precision. RoBERTa wins on recall — better at catching all AI-generated text.

---

## Files
- `fake_news_detection.py` — Mid-term project
- `ai_text_detection_bert.py` — Final project (BERT)
- `ai_text_detection_roberta.py` — Final project (RoBERTa)
