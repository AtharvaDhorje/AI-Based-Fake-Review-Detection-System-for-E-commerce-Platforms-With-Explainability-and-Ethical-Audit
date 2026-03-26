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
