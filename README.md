# HealthAI — Health Classifier

Aide à la décision clinique basée sur les habitudes de vie : estime la probabilité qu'une personne se perçoive en bonne santé à partir de 7 indicateurs déclarés (alimentation, hydratation, sommeil, activité physique).

**Live demo:** https://lea325.github.io/healthclassifier-ml/healthai.html

## What it does

The app collects lifestyle habits through a short questionnaire (diet, hydration, sleep, physical activity, snacking) and returns an instant health-risk score with clinical-style recommendations, styled as a decision-support tool for a physician.

## Model

- Logistic Regression (scikit-learn), trained on a small survey dataset (n = 19 respondents)
- Validated with Leave-One-Out cross-validation given the small sample size
- Accuracy: 84.2% · Recall on at-risk profiles: 100%

## Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn

## Files

- `app.py` — Streamlit application (model training + interactive UI)
- `healthai.html` — static version of the interface, deployed via GitHub Pages for the live demo
- `Formulaire sans titre.csv` — survey data used to train the model
- `requirements.txt` — Python dependencies

## Disclaimer

This is a portfolio / learning project. It is not a certified medical device and is not intended to replace clinical judgment.
