```markdown
# AI-Based Fake Review Detection System for E-commerce Platforms  
**With Explainability and Ethical Audit**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Stars](https://img.shields.io/github/stars/YOUR-USERNAME/AI-Fake-Review-Detection?style=social)

A complete **M.Tech project** for **AI System Design and Ethics** that detects fake vs genuine product reviews using a **leakage-safe** machine learning pipeline, Gradient Boosting, and full SHAP explainability.

## ✨ Key Features
- Strict no-leakage pipeline (split-first → train-only TF-IDF & scaling)
- 12 engineered linguistic + sentiment features
- Gradient Boosting model (best performer)
- Global + Local SHAP explanations
- Fairness analysis across rating groups (1–5 stars)
- Full ethical audit and limitations documented

## 📊 Dataset
- **Kaggle Fake Reviews Dataset** (40,432 balanced reviews)
- Labels: `CG` = Genuine (0), `OR` = Fake (1)
- Link: [https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/AI-Fake-Review-Detection.git
cd AI-Fake-Review-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebook/fake_review_detection.ipynb
```

## 📁 Project Structure
```
AI-Fake-Review-Detection/
├── README.md
├── REPORT.md
├── PRESENTATION.md
├── LICENSE
├── NOTICE
├── requirements.txt
├── notebook/
│   └── fake_review_detection.ipynb
├── images/                  # EDA plots, SHAP visualizations
└── data/                    # (optional) dataset/fake_reviews_dataset.csv
```

## 📋 Code Highlights from the Notebook

### Imports & Dataset Loading
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
nltk.download(['punkt', 'stopwords', 'vader_lexicon'], quiet=True)

df = pd.read_csv('dataset/fake_reviews_dataset.csv')
print(f"Dataset shape: {df.shape}")   # (40432, 4)
```

### Leakage-Safe Split & TF-IDF
```python
# Split BEFORE any preprocessing
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.15, random_state=42, stratify=y)

# TF-IDF fitted ONLY on train
tfidf_vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw['cleaned_text']).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw['cleaned_text']).toarray()
```

### Final Model Performance
- **Test Accuracy**: 87.02%  
- **Test F1-Score**: **86.93%**  
- **ROC-AUC**: 0.9453

## 📈 Explainable AI (SHAP)
- Global feature importance: `unique_word_ratio`, `stopword_ratio`, `text_length`, TF-IDF bigrams, `sentiment_score`
- Local explanations available in notebook (force plots)

## 📋 Rubric Compliance
- ✅ Problem Definition  
- ✅ Literature Review (18 real references)  
- ✅ Results with XAI (SHAP)  
- ✅ Full Report + PPT

## 📄 Additional Files
- `REPORT.md` → Complete project report (rubric-ready)
- `PRESENTATION.md` → 12-slide PPT outline for demo
- Full 52-page notebook with all outputs

## 📜 License
This project is licensed under the **Apache License 2.0**.  
See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.

**Copyright © 2026 Atharva Amit Dhorje**

---

**Made with ❤️ for M.Tech AI System Design and Ethics**

Feel free to star ⭐ the repo if you find it useful!
```

---

**How to use:**
1. Create a file named **`README.md`** in the root of your repository.
2. Paste the entire content above.
3. Replace `YOUR-USERNAME` with your actual GitHub username.
4. Make sure you also have the `LICENSE` (Apache 2.0) and `NOTICE` files in the same folder (I already gave you the NOTICE).
