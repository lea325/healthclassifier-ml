import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bilan Mode de Vie — Outil clinique",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #F7F8FA !important;
}
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stRadio"] > label { display: none; }

.shell {
    display: grid;
    grid-template-columns: 380px 1fr;
    min-height: 100vh;
}
.left-panel {
    background: #FFFFFF;
    border-right: 1px solid #E8EAF0;
    display: flex;
    flex-direction: column;
}
.left-top {
    padding: 32px 28px 24px;
    border-bottom: 1px solid #E8EAF0;
}
.app-name {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 8px;
}
.app-title {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    line-height: 1.3;
    margin-bottom: 4px;
}
.app-context {
    font-size: 13px;
    color: #6B7280;
    line-height: 1.5;
}
.form-body { padding: 24px 28px; flex: 1; overflow-y: auto; }
.form-section-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 16px;
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid #F3F4F6;
}
.form-section-title:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.field { margin-bottom: 18px; }
.field-label {
    font-size: 13px;
    font-weight: 500;
    color: #374151;
    margin-bottom: 8px;
    display: block;
}
.toggle {
    display: flex;
    border: 1px solid #D1D5DB;
    border-radius: 8px;
    overflow: hidden;
    background: #F9FAFB;
}
.toggle-opt {
    flex: 1;
    padding: 9px 12px;
    font-size: 13px;
    color: #6B7280;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: center;
    transition: all 0.12s;
    line-height: 1.3;
}
.toggle-opt + .toggle-opt { border-left: 1px solid #D1D5DB; }
.toggle-opt.on { background: #1B4FD8; color: #FFFFFF; font-weight: 500; }

.right-panel { padding: 40px 48px; background: #F7F8FA; }

.result-block {
    border-radius: 12px;
    padding: 32px 36px;
    margin-bottom: 28px;
    border: 1.5px solid;
}
.result-block.green  { background: #F0FDF4; border-color: #6EE7B7; }
.result-block.amber  { background: #FFFBEB; border-color: #FCD34D; }
.result-block.red    { background: #FEF2F2; border-color: #FCA5A5; }

.result-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-eyebrow.green { color: #059669; }
.result-eyebrow.amber { color: #D97706; }
.result-eyebrow.red   { color: #DC2626; }

.result-title {
    font-size: 24px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 10px;
    line-height: 1.2;
}
.result-text {
    font-size: 14px;
    color: #374151;
    line-height: 1.7;
    max-width: 580px;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 24px;
    margin-top: 24px;
    padding-top: 22px;
    border-top: 1px solid rgba(0,0,0,0.08);
}
.prob-big {
    font-size: 52px;
    font-weight: 300;
    letter-spacing: -0.03em;
    line-height: 1;
    min-width: 110px;
}
.prob-big.green { color: #059669; }
.prob-big.amber { color: #D97706; }
.prob-big.red   { color: #DC2626; }

.prob-right { flex: 1; }
.prob-label {
    font-size: 12px;
    color: #6B7280;
    margin-bottom: 8px;
}
.track {
    height: 6px;
    background: rgba(0,0,0,0.07);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 6px;
}
.fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}
.fill.green { background: #059669; }
.fill.amber { background: #D97706; }
.fill.red   { background: #DC2626; }
.track-legend {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #9CA3AF;
}

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card {
    background: #FFFFFF;
    border: 1px solid #E8EAF0;
    border-radius: 10px;
    padding: 22px 24px;
}
.card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 16px;
}
.reco-item {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #F3F4F6;
    align-items: flex-start;
}
.reco-item:last-child { border-bottom: none; }
.reco-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 6px;
}
.reco-dot.action { background: #F59E0B; }
.reco-dot.ok     { background: #10B981; }
.reco-text {
    font-size: 13px;
    color: #374151;
    line-height: 1.6;
}

.factor-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #F3F4F6;
}
.factor-item:last-child { border-bottom: none; }
.factor-icon {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 14px;
}
.factor-icon.favorable  { background: #ECFDF5; }
.factor-icon.defavorable{ background: #FEF2F2; }
.factor-icon.neutre     { background: #F3F4F6; }
.factor-name { flex: 1; font-size: 13px; color: #374151; }
.factor-tag {
    font-size: 11px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 5px;
}
.factor-tag.favorable  { background: #ECFDF5; color: #059669; }
.factor-tag.defavorable{ background: #FEF2F2; color: #DC2626; }
.factor-tag.neutre     { background: #F3F4F6; color: #9CA3AF; }

.disclaimer {
    font-size: 12px;
    color: #9CA3AF;
    line-height: 1.6;
    padding: 16px 20px;
    background: #FFFFFF;
    border: 1px solid #E8EAF0;
    border-radius: 8px;
    margin-top: 20px;
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
    return model

model = load_and_train()

def predict(f, s):
    z = -0.8624 + 1.881*f + 0.740*s
    return round(1/(1+np.exp(-z))*100, 1)

# Session state
defaults = dict(fruits=1, sport=1, water=1, sleep=1, skip=1, snacks=1, cooking=1)
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.markdown('<div class="shell">', unsafe_allow_html=True)

# ── PANNEAU GAUCHE ──
st.markdown("""
<div class="left-panel">
  <div class="left-top">
    <div class="app-name">Outil clinique</div>
    <div class="app-title">Bilan Mode de Vie</div>
    <div class="app-context">Renseignez les habitudes du patient pour obtenir une évaluation de son profil de santé.</div>
  </div>
  <div class="form-body">
""", unsafe_allow_html=True)

with st.sidebar:
    pass

# Questions via colonnes cachées — on utilise les radio Streamlit dans le main
col_hidden = st.columns([380, 1])[0]
with col_hidden:

    st.markdown('<div class="form-section-title">Alimentation</div>', unsafe_allow_html=True)

    fruits = st.radio("Fruits et légumes", ["2 portions ou plus par jour", "Moins de 2 portions par jour"], key="q_fruits")
    st.session_state.fruits = 1 if "2 portions ou" in fruits else 0

    sport = st.radio("Activité physique", ["3 séances ou plus par semaine", "Moins de 3 séances par semaine"], key="q_sport")
    st.session_state.sport = 1 if "3 séances ou" in sport else 0

    cooking = st.radio("Mode alimentaire", ["Cuisine à domicile principalement", "Plats préparés ou livraison"], key="q_cooking")
    st.session_state.cooking = 1 if "domicile" in cooking else 0

    snacks = st.radio("Grignotage", ["Absent ou très occasionnel", "Régulier entre les repas"], key="q_snacks")
    st.session_state.snacks = 1 if "Absent" in snacks else 0

    skip = st.radio("Régularité des repas", ["Repas réguliers (3 par jour)", "Repas sautés régulièrement"], key="q_skip")
    st.session_state.skip = 1 if "réguliers" in skip else 0

    st.markdown('<div class="form-section-title">Hydratation & sommeil</div>', unsafe_allow_html=True)

    water = st.radio("Hydratation", ["1,5 litre ou plus par jour", "Moins de 1,5 litre par jour"], key="q_water")
    st.session_state.water = 1 if "1,5 litre ou" in water else 0

    sleep = st.radio("Sommeil", ["7 heures ou plus par nuit", "Moins de 7 heures par nuit"], key="q_sleep")
    st.session_state.sleep = 1 if "7 heures ou" in sleep else 0

st.markdown('</div></div>', unsafe_allow_html=True)

# Calcul
f = st.session_state.fruits
s = st.session_state.sport
pct = predict(f, s)

if pct >= 65:
    cls = "green"
    eyebrow = "Profil favorable"
    titre   = "Bonne santé globale"
    texte   = ("Les habitudes de vie de ce patient sont globalement favorables à sa santé. "
               "Son alimentation et son niveau d'activité physique sont satisfaisants. "
               "Un suivi standard lors des consultations de routine est recommandé.")
elif pct >= 45:
    cls = "amber"
    eyebrow = "Profil de vigilance"
    titre   = "Ajustements recommandés"
    texte   = ("Les habitudes de vie de ce patient présentent certains facteurs de risque modérés. "
               "Des recommandations hygiéno-diététiques ciblées sont conseillées. "
               "Un suivi préventif renforcé lors des prochaines consultations est suggéré.")
else:
    cls = "red"
    eyebrow = "Profil à risque"
    titre   = "Bilan clinique recommandé"
    texte   = ("Les habitudes de vie de ce patient présentent plusieurs facteurs défavorables. "
               "Un bilan clinique approfondi est recommandé afin d'évaluer l'impact sur son état de santé. "
               "Ne pas différer la prise en charge.")

# ── PANNEAU DROIT ──
st.markdown('<div class="right-panel">', unsafe_allow_html=True)

# Résultat principal
st.markdown(f"""
<div class="result-block {cls}">
  <div class="result-eyebrow {cls}">{eyebrow}</div>
  <div class="result-title">{titre}</div>
  <div class="result-text">{texte}</div>
  <div class="prob-row">
    <div class="prob-big {cls}">{pct}<span style="font-size:22px;font-weight:400">%</span></div>
    <div class="prob-right">
      <div class="prob-label">Probabilité d'être en bonne santé selon les habitudes déclarées</div>
      <div class="track"><div class="fill {cls}" style="width:{pct}%"></div></div>
      <div class="track-legend"><span>Risque élevé</span><span>Profil favorable</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Deux colonnes
st.markdown('<div class="two-col">', unsafe_allow_html=True)

# Carte recommandations
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Recommandations pour ce patient</div>', unsafe_allow_html=True)

recos = []
if f == 0:
    recos.append("Augmenter la consommation de fruits et légumes — objectif : 2 portions minimum par jour.")
if s == 0:
    recos.append("Introduire une activité physique régulière — objectif : 3 séances de 30 minutes par semaine.")
if st.session_state.snacks == 0:
    recos.append("Réduire le grignotage entre les repas, en particulier les aliments à haute densité calorique.")
if st.session_state.water == 0:
    recos.append("Augmenter l'apport hydrique — objectif : 1,5 litre d'eau par jour minimum.")
if st.session_state.sleep == 0:
    recos.append("Améliorer l'hygiène du sommeil — objectif : 7 heures de sommeil par nuit.")
if st.session_state.cooking == 0:
    recos.append("Privilégier la cuisine à domicile pour réduire les apports en sel, sucre et graisses transformées.")
if st.session_state.skip == 0:
    recos.append("Structurer les prises alimentaires en 3 repas réguliers par jour.")

if recos:
    for r in recos:
        st.markdown(f'<div class="reco-item"><div class="reco-dot action"></div><div class="reco-text">{r}</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="reco-item"><div class="reco-dot ok"></div><div class="reco-text">Les habitudes de vie du patient sont satisfaisantes sur l\'ensemble des critères évalués. Maintenir ce mode de vie et reconduire l\'évaluation lors du prochain bilan annuel.</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Carte facteurs
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Synthèse des habitudes évaluées</div>', unsafe_allow_html=True)

items = [
    ("Alimentation · fruits & légumes", "🥗", f, "2 portions ou plus / jour", "Moins de 2 portions / jour"),
    ("Activité physique",               "🏃", s, "3 séances ou plus / semaine", "Moins de 3 séances / semaine"),
    ("Hydratation",                     "💧", st.session_state.water, "1,5 L ou plus / jour", "Moins de 1,5 L / jour"),
    ("Sommeil",                         "🌙", st.session_state.sleep, "7 h ou plus / nuit", "Moins de 7 h / nuit"),
    ("Alimentation · mode",             "🍽", st.session_state.cooking, "Cuisine à domicile", "Plats préparés / livraison"),
    ("Régularité des repas",            "🕐", st.session_state.skip, "Repas réguliers", "Repas sautés"),
    ("Grignotage",                      "🚫", st.session_state.snacks, "Absent", "Régulier"),
]

for name, icon, val, label_good, label_bad in items:
    tag_cls  = "favorable" if val == 1 else "defavorable"
    tag_txt  = label_good if val == 1 else label_bad
    icon_cls = "favorable" if val == 1 else "defavorable"
    st.markdown(f"""
    <div class="factor-item">
      <div class="factor-icon {icon_cls}">{icon}</div>
      <div class="factor-name">{name}</div>
      <div class="factor-tag {tag_cls}">{tag_txt}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
  Cet outil est un aide à la décision clinique basé sur les habitudes de vie déclarées par le patient.
  Il ne se substitue pas au jugement du praticien ni à un examen clinique complet.
  Les résultats doivent être interprétés dans le contexte global de la consultation.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)