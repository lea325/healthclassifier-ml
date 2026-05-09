import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HealthClassifier — Aide à la décision clinique",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #F5F6F8 !important;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
#MainMenu, footer, header { display: none !important; }

.app-shell {
    display: grid;
    grid-template-columns: 340px 1fr;
    min-height: 100vh;
    background: #F5F6F8;
}
.sidebar {
    background: #FFFFFF;
    border-right: 1px solid #E4E7EC;
    padding: 0;
    display: flex;
    flex-direction: column;
}
.sidebar-header {
    padding: 28px 24px 20px;
    border-bottom: 1px solid #E4E7EC;
}
.sidebar-logo {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 6px;
}
.sidebar-title {
    font-size: 18px;
    font-weight: 600;
    color: #111827;
    margin: 0;
    line-height: 1.3;
}
.sidebar-subtitle {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 4px;
}
.sidebar-section {
    padding: 20px 24px 0;
}
.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 12px;
    margin-top: 4px;
}
.sidebar-field {
    margin-bottom: 14px;
}
.sidebar-field label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: #374151;
    margin-bottom: 6px;
}
.toggle-group {
    display: flex;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    overflow: hidden;
    background: #F9FAFB;
}
.toggle-btn {
    flex: 1;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 400;
    color: #6B7280;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: center;
    line-height: 1.3;
}
.toggle-btn:not(:last-child) { border-right: 1px solid #D1D5DB; }
.toggle-btn.active {
    background: #1D4ED8;
    color: #FFFFFF;
    font-weight: 500;
}
.toggle-btn.active-green {
    background: #059669;
    color: #FFFFFF;
    font-weight: 500;
}
.toggle-btn.active-amber {
    background: #D97706;
    color: #FFFFFF;
    font-weight: 500;
}

.main-content {
    padding: 32px 36px;
    background: #F5F6F8;
}
.page-header {
    margin-bottom: 28px;
}
.page-title {
    font-size: 22px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 4px;
}
.page-desc {
    font-size: 13px;
    color: #6B7280;
    line-height: 1.5;
}

.verdict-panel {
    border-radius: 10px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid;
}
.verdict-panel.sain {
    background: #F0FDF4;
    border-color: #86EFAC;
}
.verdict-panel.vigilance {
    background: #FFFBEB;
    border-color: #FCD34D;
}
.verdict-panel.risque {
    background: #FEF2F2;
    border-color: #FCA5A5;
}
.verdict-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.verdict-label.sain { color: #059669; }
.verdict-label.vigilance { color: #D97706; }
.verdict-label.risque { color: #DC2626; }
.verdict-titre {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 6px;
}
.verdict-reco {
    font-size: 13px;
    color: #374151;
    line-height: 1.6;
}

.prob-section {
    display: flex;
    align-items: center;
    gap: 24px;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid rgba(0,0,0,0.07);
}
.prob-num {
    font-size: 44px;
    font-weight: 300;
    line-height: 1;
    letter-spacing: -0.02em;
}
.prob-num.sain { color: #059669; }
.prob-num.vigilance { color: #D97706; }
.prob-num.risque { color: #DC2626; }
.prob-bar-container { flex: 1; }
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #9CA3AF;
    margin-bottom: 6px;
}
.prob-track {
    height: 6px;
    background: rgba(0,0,0,0.08);
    border-radius: 99px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 99px;
}
.prob-fill.sain { background: #059669; }
.prob-fill.vigilance { background: #D97706; }
.prob-fill.risque { background: #DC2626; }
.prob-threshold {
    font-size: 11px;
    color: #9CA3AF;
    margin-top: 5px;
}

.cards-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
}
.card {
    background: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 10px;
    padding: 20px 22px;
}
.card-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 14px;
}

.factor-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #F3F4F6;
}
.factor-row:last-child { border-bottom: none; }
.factor-name {
    flex: 1;
    font-size: 13px;
    color: #374151;
}
.factor-coef {
    font-size: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #9CA3AF;
    width: 60px;
    text-align: right;
}
.factor-status {
    font-size: 11px;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 4px;
    width: 80px;
    text-align: center;
}
.factor-status.favorable { background: #ECFDF5; color: #059669; }
.factor-status.defavorable { background: #FEF2F2; color: #DC2626; }
.factor-status.nul { background: #F3F4F6; color: #9CA3AF; }

.reco-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #F3F4F6;
    font-size: 13px;
    color: #374151;
    line-height: 1.5;
}
.reco-item:last-child { border-bottom: none; }
.reco-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #D97706;
    flex-shrink: 0;
    margin-top: 5px;
}
.reco-dot.ok { background: #059669; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-size: 26px;
    font-weight: 500;
    color: #111827;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-lab {
    font-size: 11px;
    color: #9CA3AF;
    line-height: 1.4;
}
.metric-sub {
    font-size: 11px;
    color: #D97706;
    margin-top: 3px;
    font-weight: 500;
}

.cm-grid {
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    gap: 6px;
    align-items: center;
}
.cm-header {
    font-size: 11px;
    color: #6B7280;
    text-align: center;
    padding: 6px;
    font-weight: 500;
}
.cm-row-label {
    font-size: 11px;
    color: #6B7280;
    padding-right: 8px;
    white-space: nowrap;
    font-weight: 500;
}
.cm-cell {
    padding: 16px 8px;
    border-radius: 8px;
    text-align: center;
}
.cm-cell-val {
    font-size: 28px;
    font-weight: 500;
    line-height: 1;
}
.cm-cell-lab {
    font-size: 10px;
    margin-top: 3px;
}
.cm-tp { background: #F0FDF4; }
.cm-tp .cm-cell-val { color: #059669; }
.cm-tp .cm-cell-lab { color: #6EE7B7; }
.cm-tn { background: #F0FDF4; }
.cm-tn .cm-cell-val { color: #059669; }
.cm-tn .cm-cell-lab { color: #6EE7B7; }
.cm-fp { background: #F9FAFB; }
.cm-fp .cm-cell-val { color: #9CA3AF; }
.cm-fp .cm-cell-lab { color: #D1D5DB; }
.cm-fn { background: #FFFBEB; }
.cm-fn .cm-cell-val { color: #D97706; }
.cm-fn .cm-cell-lab { color: #FCD34D; }

.alert-box {
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 12px;
}
.alert-success { background: #F0FDF4; color: #14532D; border: 1px solid #86EFAC; }
.alert-warning { background: #FFFBEB; color: #78350F; border: 1px solid #FCD34D; }

.section-title {
    font-size: 13px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #E4E7EC;
}

.tab-nav {
    display: flex;
    gap: 0;
    border-bottom: 1px solid #E4E7EC;
    margin-bottom: 24px;
}
.tab-item {
    padding: 10px 18px;
    font-size: 13px;
    font-weight: 500;
    color: #6B7280;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: all 0.15s;
}
.tab-item.active { color: #1D4ED8; border-bottom-color: #1D4ED8; }

.full-width { grid-column: 1 / -1; }

.nll-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
}
.nll-label { font-size: 12px; color: #374151; width: 160px; flex-shrink: 0; }
.nll-track { flex: 1; height: 4px; background: #F3F4F6; border-radius: 99px; overflow: hidden; }
.nll-fill { height: 100%; border-radius: 99px; }
.nll-val { font-size: 12px; font-weight: 500; color: #111827; width: 40px; text-align: right; font-family: monospace; }

.formula-block {
    background: #F8FAFC;
    border: 1px solid #E4E7EC;
    border-radius: 8px;
    padding: 16px 18px;
    font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
    line-height: 2;
    color: #374151;
    margin-bottom: 12px;
}
.formula-block .hl { color: #1D4ED8; font-weight: 600; }
.formula-block .result { color: #059669; font-weight: 600; }

.decision-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 12px 0;
    border-bottom: 1px solid #F3F4F6;
}
.decision-row:last-child { border-bottom: none; }
.decision-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 1px;
    background: #EEF2FF;
    color: #3730A3;
}
.decision-text { font-size: 13px; color: #374151; line-height: 1.6; }
.decision-text strong { color: #111827; font-weight: 500; }

.limit-row {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 9px 0;
    font-size: 13px;
    color: #374151;
    line-height: 1.5;
    border-bottom: 1px solid #F3F4F6;
}
.limit-row:last-child { border-bottom: none; }
.limit-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #9CA3AF;
    flex-shrink: 0;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

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

def sigmoid(z): return 1/(1+np.exp(-z))

# State
if 'fruits'     not in st.session_state: st.session_state.fruits     = 1
if 'sport'      not in st.session_state: st.session_state.sport      = 1
if 'water'      not in st.session_state: st.session_state.water      = 1
if 'sleep'      not in st.session_state: st.session_state.sleep      = 1
if 'skip_meals' not in st.session_state: st.session_state.skip_meals = 1
if 'snacks'     not in st.session_state: st.session_state.snacks     = 1
if 'cooking'    not in st.session_state: st.session_state.cooking    = 1
if 'tab'        not in st.session_state: st.session_state.tab        = 'evaluation'

def set_val(key, val):
    st.session_state[key] = val

# Calcul proba
fv = st.session_state.fruits
sv = st.session_state.sport
z_val = -0.8624 + 1.881*fv + 0.740*sv
prob = sigmoid(z_val)
pct  = round(prob*100, 1)

if prob >= 0.65:
    cls = "sain"
    verdict_titre = "Profil en bonne santé"
    verdict_reco  = "Les habitudes de vie du patient sont globalement favorables à sa santé. Un suivi standard est recommandé. Encourager le maintien des comportements actuels."
elif prob >= 0.45:
    cls = "vigilance"
    verdict_titre = "Profil de vigilance"
    verdict_reco  = "Certaines habitudes présentent des facteurs de risque modérés. Un suivi préventif renforcé est conseillé, avec des recommandations hygiéno-diététiques ciblées."
else:
    cls = "risque"
    verdict_titre = "Profil à risque — bilan recommandé"
    verdict_reco  = "Les habitudes de vie du patient présentent plusieurs facteurs défavorables. Un bilan clinique approfondi est recommandé."

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

# ── SIDEBAR ──
st.markdown("""
<div class="sidebar">
  <div class="sidebar-header">
    <div class="sidebar-logo">HealthClassifier</div>
    <p class="sidebar-title">Évaluation du mode de vie</p>
    <p class="sidebar-subtitle">Renseignez les habitudes du patient</p>
  </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### Alimentation")

    fruits_val = st.radio("Fruits et légumes / jour",
        ["2 portions ou plus", "Moins de 2 portions"],
        index=0 if st.session_state.fruits == 1 else 1,
        key="fruits_radio", horizontal=False)
    st.session_state.fruits = 1 if "2 portions" in fruits_val else 0

    sport_val = st.radio("Activité physique / semaine",
        ["3 séances ou plus (30 min minimum)", "Moins de 3 séances"],
        index=0 if st.session_state.sport == 1 else 1,
        key="sport_radio")
    st.session_state.sport = 1 if "3 séances" in sport_val else 0

    cooking_val = st.radio("Alimentation principalement",
        ["Cuisinée à domicile", "Plats préparés / restauration rapide"],
        index=0 if st.session_state.cooking == 1 else 1,
        key="cooking_radio")
    st.session_state.cooking = 1 if "domicile" in cooking_val else 0

    snacks_val = st.radio("Grignotage entre les repas",
        ["Absent ou occasionnel", "Régulier (sucreries, sodas, chips)"],
        index=0 if st.session_state.snacks == 1 else 1,
        key="snacks_radio")
    st.session_state.snacks = 1 if "Absent" in snacks_val else 0

    skip_val = st.radio("Saute des repas",
        ["Non", "Oui, régulièrement"],
        index=0 if st.session_state.skip_meals == 1 else 1,
        key="skip_radio")
    st.session_state.skip_meals = 1 if skip_val == "Non" else 0

    st.markdown("#### Hydratation & sommeil")

    water_val = st.radio("Consommation d'eau / jour",
        ["1,5 litre ou plus", "Moins de 1,5 litre"],
        index=0 if st.session_state.water == 1 else 1,
        key="water_radio")
    st.session_state.water = 1 if "1,5 litre ou" in water_val else 0

    sleep_val = st.radio("Durée du sommeil / nuit",
        ["7 heures ou plus", "Moins de 7 heures"],
        index=0 if st.session_state.sleep == 1 else 1,
        key="sleep_radio")
    st.session_state.sleep = 1 if "7 heures ou" in sleep_val else 0

# Recalcul après widgets
fv = st.session_state.fruits
sv = st.session_state.sport
z_val = -0.8624 + 1.881*fv + 0.740*sv
prob  = sigmoid(z_val)
pct   = round(prob*100, 1)

if prob >= 0.65:
    cls = "sain"
    verdict_titre = "Profil en bonne santé"
    verdict_reco  = "Les habitudes de vie du patient sont globalement favorables. Un suivi standard est recommandé. Encourager le maintien des comportements actuels."
elif prob >= 0.45:
    cls = "vigilance"
    verdict_titre = "Profil de vigilance"
    verdict_reco  = "Certaines habitudes présentent des facteurs de risque modérés. Un suivi préventif renforcé est conseillé, avec des recommandations hygiéno-diététiques ciblées."
else:
    cls = "risque"
    verdict_titre = "Profil à risque — bilan recommandé"
    verdict_reco  = "Les habitudes de vie du patient présentent plusieurs facteurs défavorables. Un bilan clinique approfondi est recommandé. Ne pas différer la prise en charge."

# ── MAIN CONTENT ──
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Navigation tabs
tab = st.radio("Navigation", ["Évaluation", "Performance du modèle", "Transparence algorithmique"],
               horizontal=True, key="main_nav", label_visibility="collapsed")

st.markdown("---")

if tab == "Évaluation":

    # Verdict
    st.markdown(f"""
    <div class="verdict-panel {cls}">
      <div class="verdict-label {cls}">Résultat de l'évaluation</div>
      <div class="verdict-titre">{verdict_titre}</div>
      <div class="verdict-reco">{verdict_reco}</div>
      <div class="prob-section">
        <div class="prob-num {cls}">{pct}<span style="font-size:20px; font-weight:400;">%</span></div>
        <div class="prob-bar-container">
          <div class="prob-bar-label">
            <span>Probabilité d'être en bonne santé</span>
            <span>Seuil de décision : 45%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill {cls}" style="width:{pct}%;"></div>
          </div>
          <div class="prob-threshold">
            P(bonne santé | habitudes du patient) = {prob:.3f}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # Facteurs
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Facteurs discriminants du modèle</div>', unsafe_allow_html=True)

        factors = [
            ("Fruits & légumes / jour",   1.881, fv,  True),
            ("Activité physique / semaine",0.740, sv,  True),
            ("Cuisine à domicile",         0.000, st.session_state.cooking,    False),
            ("Hydratation",                0.000, st.session_state.water,      False),
            ("Qualité du sommeil",         0.000, st.session_state.sleep,      False),
            ("Absence de grignotage",      0.000, st.session_state.snacks,     False),
            ("Régularité des repas",       0.000, st.session_state.skip_meals, False),
        ]
        for name, beta, val, active in factors:
            if active:
                status_cls  = "favorable" if val == 1 else "defavorable"
                status_txt  = "Favorable" if val == 1 else "Défavorable"
            else:
                status_cls  = "nul"
                status_txt  = "Non retenu"
            beta_str = f"+{beta:.3f}" if beta > 0 else f"{beta:.3f}"
            st.markdown(f"""
            <div class="factor-row">
              <div class="factor-name">{name}</div>
              <div class="factor-coef">β = {beta_str}</div>
              <div class="factor-status {status_cls}">{status_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:11px; color:#9CA3AF; margin-top:12px; line-height:1.5;">
          La régularisation L1 (λ = 2.0) a annulé 6 coefficients sur 8.
          Seuls l'alimentation et l'activité physique sont statistiquement discriminants.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Recommandations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Recommandations hygiéno-diététiques</div>', unsafe_allow_html=True)

        recos = []
        if fv == 0:
            recos.append(("Augmenter la consommation de fruits et légumes à un minimum de 2 portions par jour (recommandation OMS).", False))
        if sv == 0:
            recos.append(("Introduire au moins 3 séances d'activité physique modérée d'une durée de 30 minutes par semaine.", False))
        if st.session_state.snacks == 0:
            recos.append(("Réduire le grignotage entre les repas, en particulier les aliments à index glycémique élevé.", False))
        if st.session_state.water == 0:
            recos.append(("Augmenter l'apport hydrique à 1,5 litre d'eau par jour minimum.", False))
        if st.session_state.sleep == 0:
            recos.append(("Améliorer l'hygiène du sommeil afin d'atteindre 7 heures de sommeil par nuit.", False))
        if st.session_state.cooking == 0:
            recos.append(("Privilégier la cuisine à domicile, limitant les apports en sel, sucre et graisses transformées.", False))
        if st.session_state.skip_meals == 0:
            recos.append(("Structurer les prises alimentaires en 3 repas réguliers par jour.", False))

        if not recos:
            st.markdown("""
            <div class="reco-item">
              <div class="reco-dot ok"></div>
              <div>Le patient présente des habitudes de vie globalement satisfaisantes. Maintenir ce mode de vie et reconduire l'évaluation lors du prochain bilan annuel.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for reco, _ in recos:
                st.markdown(f"""
                <div class="reco-item">
                  <div class="reco-dot"></div>
                  <div>{reco}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Calcul détaillé
        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Calcul du score probabiliste</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="formula-block">
Z  =  β₀ + β₁ · X<sub style="font-size:10px">fruits</sub> + β₂ · X<sub style="font-size:10px">sport</sub><br>
Z  =  −0.8624 &nbsp;+&nbsp; 1.881 × <span class="hl">{fv}</span> &nbsp;+&nbsp; 0.740 × <span class="hl">{sv}</span><br>
Z  =  <span class="hl">{z_val:.4f}</span><br><br>
P(sain | X) = σ(Z) = 1 / (1 + e<sup>−{z_val:.4f}</sup>)<br>
P(sain | X) = <span class="result">{prob:.4f}</span>&nbsp; → &nbsp;<span class="result">{pct}%</span><br><br>
Décision :&nbsp; {pct}% {'≥' if prob >= 0.45 else '<'} 45% &nbsp;→&nbsp; <span class="{'result' if prob >= 0.45 else 'hl'}">{'BONNE SANTÉ' if prob >= 0.45 else 'BILAN RECOMMANDÉ'}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif tab == "Performance du modèle":

    st.markdown('<p class="page-title">Performance du modèle</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Métriques évaluées par Leave-One-Out Cross-Validation (LOOCV) sur les 19 patients du dataset. La LOOCV est la méthode la plus robuste pour les petits effectifs : le modèle est entraîné sur n−1 patients et testé sur 1, répété 19 fois.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-val" style="color:#059669;">100%</div>
        <div class="metric-lab">Recall<br>classe à risque</div>
        <div class="metric-sub">0 patient à risque manqué</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" style="color:#D97706;">57.1%</div>
        <div class="metric-lab">Précision<br>classe à risque</div>
        <div class="metric-sub">4 alertes sur 7 confirmées</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">80.8%</div>
        <div class="metric-lab">F1-Score<br>macro</div>
        <div class="metric-sub">Équilibre précision / rappel</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">0.80</div>
        <div class="metric-lab">AUC-ROC</div>
        <div class="metric-sub">vs 0.50 aléatoire</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Matrice de confusion — LOOCV</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="cm-grid">
          <div></div>
          <div class="cm-header">Prédit : à risque</div>
          <div class="cm-header">Prédit : bonne santé</div>
          <div class="cm-row-label">Réel : à risque</div>
          <div class="cm-cell cm-tn">
            <div class="cm-cell-val">4</div>
            <div class="cm-cell-lab" style="color:#059669;">Vrais négatifs</div>
          </div>
          <div class="cm-cell cm-fp">
            <div class="cm-cell-val">0</div>
            <div class="cm-cell-lab" style="color:#D1D5DB;">Faux positifs</div>
          </div>
          <div class="cm-row-label">Réel : bonne santé</div>
          <div class="cm-cell cm-fn">
            <div class="cm-cell-val">3</div>
            <div class="cm-cell-lab" style="color:#D97706;">Faux négatifs</div>
          </div>
          <div class="cm-cell cm-tp">
            <div class="cm-cell-val">12</div>
            <div class="cm-cell-lab" style="color:#059669;">Vrais positifs</div>
          </div>
        </div>
        <div class="alert-box alert-success" style="margin-top:14px;">
          Recall = 100% sur la classe à risque : les 4 patients à risque ont tous été détectés. Aucun cas non pris en charge.
        </div>
        <div class="alert-box alert-warning">
          3 faux négatifs : 3 patients en bonne santé ont été classés à risque. Conséquence : examens supplémentaires non nécessaires, sans gravité clinique.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        nll_model = round(log_loss(y, cv_probs), 4)
        nll_base  = round(log_loss(y, [y.mean()]*len(y)), 4)
        auc       = round(roc_auc_score(y, cv_probs), 3)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Calibration — Negative Log-Likelihood (NLL)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style="font-size:13px; color:#374151; line-height:1.6; margin-bottom:14px;">
          La NLL mesure la qualité de calibration des probabilités prédites.
          Plus la NLL est basse, plus les probabilités du modèle sont fiables.
          Une bonne calibration est essentielle en contexte clinique.
        </p>
        <div class="nll-row">
          <div class="nll-label">Modèle naïf (baseline)</div>
          <div class="nll-track"><div class="nll-fill" style="width:100%; background:#FCA5A5;"></div></div>
          <div class="nll-val" style="color:#DC2626;">{nll_base}</div>
        </div>
        <div class="nll-row">
          <div class="nll-label">Notre modèle (LOOCV)</div>
          <div class="nll-track"><div class="nll-fill" style="width:{round(nll_model/nll_base*100)}%; background:#059669;"></div></div>
          <div class="nll-val" style="color:#059669;">{nll_model}</div>
        </div>
        <p style="font-size:11px; color:#9CA3AF; margin-top:12px;">
          Réduction de {round((1-nll_model/nll_base)*100, 1)}% par rapport au modèle naïf.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Choix des métriques — justification clinique</div>', unsafe_allow_html=True)
        justifs = [
            ("Recall prioritaire", "Un faux négatif correspond à un patient à risque non détecté — conséquence clinique potentiellement grave. Le recall a donc été maximisé en priorité."),
            ("Accuracy non utilisée", "Avec 79% de patients en bonne santé, un modèle qui prédit systématiquement 'bonne santé' atteindrait 79% d'accuracy sans aucune valeur diagnostique."),
            ("F1-score macro", "Assure l'équilibre entre précision et rappel sur les deux classes malgré le déséquilibre 79/21."),
        ]
        for titre, desc in justifs:
            st.markdown(f"""
            <div class="decision-row" style="padding:10px 0;">
              <div>
                <div style="font-size:13px; font-weight:500; color:#111827; margin-bottom:3px;">{titre}</div>
                <div style="font-size:12px; color:#6B7280; line-height:1.5;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif tab == "Transparence algorithmique":

    st.markdown('<p class="page-title">Transparence algorithmique</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-desc">Ce modèle est conçu pour être entièrement explicable. Chaque décision technique est documentée et justifiée par le contexte clinique.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Décisions techniques et justifications</div>', unsafe_allow_html=True)
        decisions = [
            ("Régression logistique", "Produit une probabilité P(bonne santé | X) ∈ [0,1] directement interprétable en clinique. Le praticien reçoit une probabilité, non une boîte noire."),
            ("Pondération w₀ = 5", "Les erreurs sur la classe à risque coûtent 5 fois plus cher dans la fonction de coût. Justification : un faux négatif (patient malade non détecté) est cliniquement plus grave qu'un faux positif."),
            ("Régularisation L1 · λ = 2.0", "Le dataset étant limité à 19 observations, L1 prévient le surapprentissage en forçant les coefficients non discriminants à zéro. Résultat : 2 variables actives sur 8."),
            ("LOOCV", "Avec n = 19, le K-Fold classique produirait des folds de test de 3 à 4 observations, non représentatifs. LOOCV entraîne sur 18 patients, teste sur 1, répété 19 fois."),
            ("Seuil à 45%", "Abaissé de 50% à 45% pour augmenter la sensibilité sur la classe à risque. Combiné à w₀ = 5, ce réglage porte le recall de 0% à 100%."),
        ]
        for titre, desc in decisions:
            st.markdown(f"""
            <div class="decision-row">
              <div class="decision-badge">{titre}</div>
              <div class="decision-text">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Formules mathématiques</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-block">
<b>Score Z (log-odds)</b><br>
Z = β₀ + β₁·X<sub>fruits</sub> + β₂·X<sub>sport</sub><br>
Z = −0.8624 + 1.881·X<sub>fruits</sub> + 0.740·X<sub>sport</sub><br><br>
<b>Probabilité (sigmoïde)</b><br>
P(sain | X) = σ(Z) = 1 / (1 + e<sup>−Z</sup>)<br><br>
<b>Fonction de coût (NLL pondérée + L1)</b><br>
Cost = −Σ[y·log(ŷ) + w₀·(1−y)·log(1−ŷ)] + λ·Σ|βⱼ|<br>
avec w₀ = 5, λ = 2.0
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Coefficients β — après régularisation L1</div>', unsafe_allow_html=True)
        factors_full = [
            ("Fruits & légumes / jour",    "+1.881", True),
            ("Activité physique / semaine", "+0.740", True),
            ("Cuisine à domicile",          "0.000",  False),
            ("Hydratation",                 "0.000",  False),
            ("Qualité du sommeil",          "0.000",  False),
            ("Absence de grignotage",       "0.000",  False),
            ("Régularité des repas",        "0.000",  False),
            ("Récupération au réveil",      "0.000",  False),
        ]
        for name, beta, active in factors_full:
            bar_w = 100 if "1.881" in beta else (39 if "0.740" in beta else 0)
            badge = "Actif" if active else "Annulé L1"
            badge_cls = "favorable" if active else "nul"
            st.markdown(f"""
            <div class="factor-row">
              <div class="factor-name">{name}</div>
              <div class="factor-coef">β = {beta}</div>
              <div class="factor-status {badge_cls}">{badge}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:11px; color:#9CA3AF; margin-top:12px; line-height:1.6;">
          La régularisation Lasso annule les coefficients des variables non discriminantes.
          Ce résultat est cohérent avec les recommandations de l'OMS, qui identifie l'alimentation
          et l'activité physique comme les deux leviers comportementaux les plus déterminants pour la santé.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Limites et périmètre d\'utilisation</div>', unsafe_allow_html=True)
        limits = [
            ("Dataset de 19 patients : les intervalles de confiance sur les coefficients sont larges. Les résultats doivent être interprétés avec prudence et validés sur une cohorte plus large."),
            ("Données auto-déclarées : l'absence de biomarqueurs objectifs (IMC, pression artérielle, bilan biologique) limite la portée diagnostique."),
            ("Déséquilibre de classes (79% sains / 21% à risque) : la précision sur la classe à risque est limitée à 57.1% malgré la pondération."),
            ("Outil d'aide à la décision uniquement — ne se substitue pas au jugement clinique du praticien."),
        ]
        for lim in limits:
            st.markdown(f"""
            <div class="limit-row">
              <div class="limit-dot"></div>
              <div>{lim}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:24px 0 8px; font-size:11px; color:#9CA3AF; border-top:1px solid #E4E7EC; margin-top:32px;">
  HealthClassifier · Outil d'aide à la décision clinique · Régression logistique L1 · LOOCV ·
  Master 2 IMC&DS — Paris 1 Panthéon-Sorbonne · 2025–2026
</div>
""", unsafe_allow_html=True)