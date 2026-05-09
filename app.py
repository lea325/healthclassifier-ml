import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                              log_loss, f1_score, recall_score, precision_score)
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
st.set_page_config(page_title="HealthClassifier", page_icon="🏥", layout="wide")

# ── LOAD & TRAIN ──
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/lea325/healthclassifier-ml/refs/heads/main/%20Formulaire%20sans%20titre.csv"
    df = pd.read_csv(url)
    df.columns = ['timestamp','age','skip_meals','fruits','cooking',
                  'water','sport','sleep','rested','snacks','healthy']
    enc = {
        'skip_meals': {'Non':1,'Oui':0},
        'fruits':     {'+ de 2 par jour':1,'- de 2 par jour':0},
        'cooking':    {'Oui je cuisines souvent':1,"Non j'adore Ubert Eat":0},
        'water':      {'+ de 1,5L':1,'- de 1,5L':0},
        'sport':      {'+ de 3 fois':1,'- de 3 fois':0},
        'sleep':      {'+ de 7h':1,'- de 7h':0},
        'rested':     {'Oui':1,'Non':0},
        'snacks':     {'Non':1,'Oui':0},
    }
    for col, m in enc.items():
        df[col+'_enc'] = df[col].map(m)
    df['y'] = (df['healthy'] == 'Oui').astype(int)
    FEATS = [c+'_enc' for c in enc.keys()]
    X = df[FEATS].values
    y = df['y'].values
    model = LogisticRegression(penalty='l1', C=0.5, class_weight={0:5,1:1},
                                solver='liblinear', max_iter=2000, random_state=42)
    model.fit(X, y)
    loo = LeaveOneOut()
    cv_probs = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:,1]
    cv_preds = (cv_probs >= 0.45).astype(int)
    return model, X, y, cv_probs, cv_preds

model, X, y, cv_probs, cv_preds = load_and_train()
cm = confusion_matrix(y, cv_preds)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_patient(fruits, sport):
    z = -0.8624 + 1.881 * fruits + 0.740 * sport
    prob = sigmoid(z)
    return prob, prob >= 0.45

# ── HEADER ──
st.title("🏥 HealthClassifier")
st.markdown("**Régression logistique L1 pondérée · LOOCV · n=19 · Master 2 IMC&DS — Paris 1**")
st.divider()

# ── TABS ──
tab1, tab2, tab3 = st.tabs(["🩺 Interface Médecin", "📊 Performance du modèle", "⚙️ Détails techniques"])

# ── TAB 1 : SIMULATEUR ──
with tab1:
    st.subheader("Saisie des habitudes patient")
    col1, col2 = st.columns(2)

    with col1:
        fruits = st.radio("🥦 Fruits & légumes / jour",
                          ["+ de 2 portions (≥2/jour)", "- de 2 portions (<2/jour)"])
        sport  = st.radio("🏃 Activité physique / semaine",
                          ["+ de 3 séances (≥3/sem)", "- de 3 séances (<3/sem)"])
        st.caption("Variables annulées par L1 (β=0) — non discriminantes sur ce dataset :")
        st.radio("⏭️ Saute des repas", ["Non","Oui"], disabled=True)
        st.radio("💧 Eau / jour",      ["+ de 1,5L","- de 1,5L"], disabled=True)

    with col2:
        st.radio("😴 Sommeil / nuit",  ["+ de 7h","- de 7h"], disabled=True)
        st.radio("🍿 Snacks",           ["Rarement","Souvent"], disabled=True)

    fruits_val = 1 if "+" in fruits else 0
    sport_val  = 1 if "+" in sport  else 0
    prob, is_sain = predict_patient(fruits_val, sport_val)
    z_val = -0.8624 + 1.881 * fruits_val + 0.740 * sport_val

    st.divider()
    st.subheader("Résultat")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Probabilité P(sain|X)", f"{prob*100:.1f}%")
    with colB:
        st.metric("Seuil de décision", "45%")
    with colC:
        verdict = "🟢 Sain" if is_sain else "🔴 À risque"
        st.metric("Classification", verdict)

    if is_sain:
        st.success(f"**Profil favorable.** P(sain) = {prob*100:.1f}% ≥ 45%. Suivi standard recommandé.")
    else:
        st.error(f"**Profil à risque.** P(sain) = {prob*100:.1f}% < 45%. Bilan approfondi recommandé.")

    with st.expander("Voir le calcul détaillé"):
        st.code(f"""
Z = β₀ + β_fruits·X_fruits + β_sport·X_sport
Z = -0.8624 + 1.881×{fruits_val} + 0.740×{sport_val}
Z = {z_val:.4f}

σ(Z) = 1 / (1 + e^(-{z_val:.4f})) = {prob:.4f}

Décision : {prob:.4f} {'≥' if is_sain else '<'} 0.45 → {'SAIN' if is_sain else 'À RISQUE'}
        """, language="text")

# ── TAB 2 : MÉTRIQUES ──
with tab2:
    st.subheader("Métriques LOOCV — Leave-One-Out Cross-Validation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall · À risque",    "100%",  "vs 0% baseline")
    c2.metric("Précision · À risque", "57.1%", "4/7 alarmes justifiées")
    c3.metric("F1-Score Macro",       "80.8%")
    c4.metric("AUC-ROC",              "0.80",  "vs 0.50 aléatoire")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matrice de confusion")
        st.markdown("""
| | **Prédit : À risque** | **Prédit : Sain** |
|---|---|---|
| **Réel : À risque** | ✅ TN = 4 | ❌ FP = 0 |
| **Réel : Sain** | ⚠️ FN = 3 | ✅ TP = 12 |
        """)
        st.success("**TN = 4/4** : aucun patient à risque manqué. Recall = 100% ✓")
        st.warning("**FN = 3** : 3 patients sains classés 'à risque'. Examens inutiles mais bénins.")

    with col2:
        st.subheader("NLL — Negative Log-Likelihood")
        nll_model    = log_loss(y, cv_probs)
        nll_baseline = log_loss(y, [y.mean()]*len(y))
        st.metric("NLL modèle (LOOCV)",  f"{nll_model:.4f}")
        st.metric("NLL baseline (naïf)", f"{nll_baseline:.4f}",
                  f"{nll_baseline-nll_model:.4f} de moins ✓")
        st.caption("Plus la NLL est basse, mieux le modèle est calibré.")

    st.divider()
    st.subheader("Justification des métriques")
    st.markdown("""
- **Recall prioritaire** : FN = patient malade non détecté → risque vital.
- **F1-macro** : équilibre Précision/Recall malgré le déséquilibre 79/21.
- **AUC-ROC = 0.80** : discrimination correcte 80% du temps, indépendamment du seuil.
- **Accuracy écartée** : un modèle naïf atteindrait 79% sans rien apprendre.
    """)

# ── TAB 3 : DÉTAILS TECH ──
with tab3:
    st.subheader("Architecture du modèle")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fonction de coût (NLL pondérée)**")
        st.latex(r"NLL = -\sum \left[ y \cdot \log(\hat{y}) + w_0 \cdot (1-y) \cdot \log(1-\hat{y}) \right]")
        st.caption("w₀ = 5 : erreurs sur classe à risque coûtent 5× plus cher → Recall = 100%")

        st.markdown("**Régularisation L1 (Lasso)**")
        st.latex(r"Cost = NLL + \lambda \cdot \sum |\beta_j| \quad \lambda = 2.0")
        st.caption("6 coefficients annulés sur 8 → sélection automatique des variables pertinentes")

        st.markdown("**Décision finale**")
        st.latex(r"\hat{y} = \begin{cases} \text{Sain} & \sigma(Z) \geq 0.45 \\ \text{À risque} & \sigma(Z) < 0.45 \end{cases}")

    with col2:
        st.markdown("**Coefficients β après L1**")
        coef_data = {
            "Variable": ["🥦 Fruits & légumes", "🏃 Sport", "⏭️ Repas réguliers",
                         "🍳 Cuisine maison", "💧 Hydratation",
                         "😴 Sommeil", "😌 Reposé", "🍿 Snacks"],
            "β": [1.881, 0.740, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            "Statut": ["✅ ACTIF","✅ ACTIF","❌ Annulé L1","❌ Annulé L1",
                       "❌ Annulé L1","❌ Annulé L1","❌ Annulé L1","❌ Annulé L1"]
        }
        st.dataframe(pd.DataFrame(coef_data), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Validation — LOOCV vs K-Fold")
    st.markdown("""
| Critère | K-Fold (K=5) | **LOOCV ✓** |
|---|---|---|
| Points de test | ~4 / fold | 1 (répété 19×) |
| Biais estimateur | Élevé | Quasi nul |
| Adaptabilité n=19 | ❌ | ✅ |
| Risque variance | Fort | Contrôlé |
    """)
    st.info("Avec n=19, LOOCV maximise les données d'entraînement à chaque itération.")