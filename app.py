import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HealthAI — Tableau de bord clinique",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Rajdhani:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #EEF4FF !important;
}
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stTextInput"] > label { display: none !important; }

/* ── SHELL ── */
.shell {
    min-height: 100vh;
    background: linear-gradient(145deg, #EEF4FF 0%, #E8F0FF 40%, #EEF8FF 100%);
    padding: 0;
}

/* ── TOP BAR ── */
.topbar {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(99,139,255,0.15);
    padding: 0 40px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon {
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, #4F7EFF, #2BBFFF);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.brand-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #4F7EFF, #2BBFFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.05em;
}
.brand-sub {
    font-size: 11px;
    color: #8FA3CC;
    margin-left: 4px;
    font-weight: 400;
    -webkit-text-fill-color: #8FA3CC;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: 16px;
}
.topbar-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(79,126,255,0.1);
    color: #4F7EFF;
    border: 1px solid rgba(79,126,255,0.2);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── CONTENT GRID ── */
.content {
    padding: 32px 40px;
    display: grid;
    grid-template-columns: 300px 1fr 280px;
    gap: 24px;
    align-items: start;
}

/* ── GLASS CARD ── */
.glass {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(79,126,255,0.08);
}
.glass-blue {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(79,126,255,0.15);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(79,126,255,0.10);
}

/* ── PATIENT CARD ── */
.patient-card { padding: 24px; margin-bottom: 16px; }
.patient-avatar {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F7EFF22, #2BBFFF33);
    border: 2px solid rgba(79,126,255,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    margin: 0 auto 14px;
}
.patient-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #1A2B5E;
    text-align: center;
    margin-bottom: 3px;
}
.patient-meta {
    font-size: 12px;
    color: #8FA3CC;
    text-align: center;
    margin-bottom: 16px;
}
.patient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,126,255,0.2), transparent);
    margin-bottom: 16px;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(79,126,255,0.06);
    font-size: 12px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #8FA3CC; font-weight: 500; }
.info-val { color: #1A2B5E; font-weight: 600; }

/* ── FORM SECTION ── */
.form-card { padding: 24px; }
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(79,126,255,0.1);
}
.section-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F7EFF, #2BBFFF);
    flex-shrink: 0;
}
.section-title-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #1A2B5E;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.q-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(79,126,255,0.06);
}
.q-row:last-child { border-bottom: none; }
.q-label {
    font-size: 13px;
    color: #374882;
    font-weight: 400;
    flex: 1;
}
.q-toggle {
    display: flex;
    border: 1px solid rgba(79,126,255,0.25);
    border-radius: 6px;
    overflow: hidden;
    flex-shrink: 0;
}
.q-opt {
    padding: 5px 14px;
    font-size: 11px;
    font-weight: 500;
    background: transparent;
    border: none;
    cursor: pointer;
    color: #8FA3CC;
    transition: all 0.12s;
}
.q-opt + .q-opt { border-left: 1px solid rgba(79,126,255,0.2); }
.q-opt.on {
    background: linear-gradient(135deg, #4F7EFF, #2BBFFF);
    color: #FFFFFF;
}
.q-opt.on-red {
    background: linear-gradient(135deg, #FF6B8A, #FF4F7E);
    color: #FFFFFF;
}

/* ── RESULT PANEL ── */
.result-panel { padding: 24px; }

.score-ring-wrap {
    width: 160px;
    height: 160px;
    margin: 0 auto 20px;
    position: relative;
}
.score-ring-svg {
    width: 160px;
    height: 160px;
    transform: rotate(-90deg);
}
.score-ring-inner {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.score-num {
    font-family: 'Rajdhani', sans-serif;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
}
.score-unit {
    font-size: 13px;
    color: #8FA3CC;
    margin-top: 2px;
}

.verdict-chip {
    text-align: center;
    margin-bottom: 16px;
}
.chip-inner {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.chip-green { background: rgba(16,185,129,0.12); color: #059669; border: 1px solid rgba(16,185,129,0.25); }
.chip-amber { background: rgba(245,158,11,0.12); color: #D97706; border: 1px solid rgba(245,158,11,0.25); }
.chip-red   { background: rgba(255,79,126,0.12); color: #DC2626; border: 1px solid rgba(255,79,126,0.25); }

.verdict-text {
    font-size: 13px;
    color: #374882;
    line-height: 1.6;
    text-align: center;
    margin-bottom: 20px;
}

/* ── MINI INDICATORS ── */
.indicators { margin-bottom: 0; }
.ind-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(79,126,255,0.06);
}
.ind-item:last-child { border-bottom: none; }
.ind-bar-wrap {
    flex: 1;
    height: 4px;
    background: rgba(79,126,255,0.1);
    border-radius: 99px;
    overflow: hidden;
}
.ind-bar {
    height: 100%;
    border-radius: 99px;
}
.ind-bar.g { background: linear-gradient(90deg, #34D399, #10B981); }
.ind-bar.r { background: linear-gradient(90deg, #FCA5A5, #F87171); }
.ind-label { font-size: 11px; color: #8FA3CC; width: 80px; flex-shrink: 0; }
.ind-val { font-size: 11px; font-weight: 600; width: 16px; flex-shrink: 0; }
.ind-val.g { color: #10B981; }
.ind-val.r { color: #DC2626; }

/* ── RECO CARD ── */
.reco-card { padding: 20px 24px; margin-top: 0; }
.reco-item {
    display: flex;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(79,126,255,0.06);
    align-items: flex-start;
}
.reco-item:last-child { border-bottom: none; }
.reco-bullet {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #4F7EFF;
    flex-shrink: 0;
    margin-top: 6px;
}
.reco-bullet.ok { background: #10B981; }
.reco-txt { font-size: 12px; color: #374882; line-height: 1.55; }

/* ── SCAN VISUAL CENTER ── */
.center-content { display: flex; flex-direction: column; gap: 16px; }
.scan-card {
    padding: 24px;
    text-align: center;
    min-height: 320px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}
.scan-rings {
    position: absolute;
    width: 280px;
    height: 280px;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
}
.scan-figure {
    position: relative;
    z-index: 2;
    font-size: 96px;
    line-height: 1;
    filter: drop-shadow(0 0 20px rgba(79,126,255,0.3));
}
.scan-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #4F7EFF;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 12px;
    position: relative;
    z-index: 2;
}

/* ── STAT CHIPS ── */
.stat-chips {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
}
.stat-chip {
    padding: 16px;
    border-radius: 12px;
    text-align: center;
}
.stat-chip-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 24px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-chip-lab {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.75;
}
.chip-1 { background: rgba(79,126,255,0.08); border: 1px solid rgba(79,126,255,0.15); color: #4F7EFF; }
.chip-2 { background: rgba(43,191,255,0.08); border: 1px solid rgba(43,191,255,0.15); color: #2BBFFF; }
.chip-3 { background: rgba(99,91,255,0.08);  border: 1px solid rgba(99,91,255,0.15);  color: #635BFF; }

.disclaimer-text {
    font-size: 11px;
    color: #8FA3CC;
    text-align: center;
    padding: 12px 16px;
    border-top: 1px solid rgba(79,126,255,0.08);
    line-height: 1.5;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
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
    for c,m in enc.items(): df[c+'_enc'] = df[c].map(m)
    df['y'] = (df['healthy']=='Oui').astype(int)
    X = df[[c+'_enc' for c in enc]].values
    y = df['y'].values
    mod = LogisticRegression(penalty='l1',C=0.5,class_weight={0:5,1:1},
                             solver='liblinear',max_iter=2000,random_state=42)
    mod.fit(X, y)
    return mod

model = load_model()

def get_prob(f,s):
    return round(1/(1+np.exp(-(-0.8624+1.881*f+0.740*s)))*100, 1)

# Session defaults
for k,v in dict(fruits=1,sport=1,water=1,sleep=1,skip=1,snacks=1,cooking=1).items():
    if k not in st.session_state: st.session_state[k] = v

# ── TOP BAR ──
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="brand-icon">✦</div>
    <span class="brand-name">HealthAI</span>
    <span class="brand-sub">Système d'aide à la décision clinique</span>
  </div>
  <div class="topbar-right">
    <span class="topbar-badge">Dr. Mode</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="shell"><div class="content">', unsafe_allow_html=True)

# ════════════════════════════════════
# COLONNE GAUCHE — PATIENT + FORM
# ════════════════════════════════════
st.markdown('<div style="display:flex;flex-direction:column;gap:16px;">', unsafe_allow_html=True)

# Identité patient
st.markdown('<div class="glass patient-card">', unsafe_allow_html=True)
prenom = st.text_input("Prénom", value="Marie", placeholder="Prénom", label_visibility="collapsed")
nom    = st.text_input("Nom", value="Dupont", placeholder="Nom de famille", label_visibility="collapsed")
age    = st.number_input("Âge", min_value=1, max_value=120, value=42, label_visibility="collapsed")
medecin= st.text_input("Médecin", value="Dr. Martin", placeholder="Médecin traitant", label_visibility="collapsed")

initiales = (prenom[0] if prenom else "?") + (nom[0] if nom else "?")
st.markdown(f"""
<div class="patient-avatar">{initiales.upper()}</div>
<div class="patient-name">{prenom.upper()} {nom.upper()}</div>
<div class="patient-meta">{age} ans · Patient(e)</div>
<div class="patient-divider"></div>
<div class="info-row"><span class="info-key">Médecin</span><span class="info-val">{medecin}</span></div>
<div class="info-row"><span class="info-key">Évaluation</span><span class="info-val">Mode de vie</span></div>
<div class="info-row"><span class="info-key">Modèle IA</span><span class="info-val">HealthAI v1.0</span></div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Formulaire
st.markdown('<div class="glass form-card">', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
  <div class="section-dot"></div>
  <div class="section-title-text">Habitudes alimentaires</div>
</div>
""", unsafe_allow_html=True)

fruits_r = st.radio("f", ["≥ 2 portions/jour","< 2 portions/jour"], horizontal=True, key="rf")
st.session_state.fruits = 1 if "≥ 2" in fruits_r else 0

sport_r = st.radio("s", ["≥ 3 séances/sem","< 3 séances/sem"], horizontal=True, key="rs")
st.session_state.sport = 1 if "≥ 3" in sport_r else 0

cooking_r = st.radio("c", ["Cuisine maison","Plats préparés"], horizontal=True, key="rc")
st.session_state.cooking = 1 if "maison" in cooking_r else 0

snacks_r = st.radio("sn", ["Absent","Régulier"], horizontal=True, key="rsn")
st.session_state.snacks = 1 if "Absent" in snacks_r else 0

skip_r = st.radio("sk", ["Repas réguliers","Repas sautés"], horizontal=True, key="rsk")
st.session_state.skip = 1 if "réguliers" in skip_r else 0

st.markdown("""
<div class="section-header" style="margin-top:20px;">
  <div class="section-dot"></div>
  <div class="section-title-text">Hydratation & Sommeil</div>
</div>
""", unsafe_allow_html=True)

water_r = st.radio("w", ["≥ 1,5 L/jour","< 1,5 L/jour"], horizontal=True, key="rw")
st.session_state.water = 1 if "≥ 1,5" in water_r else 0

sleep_r = st.radio("sl", ["≥ 7 h/nuit","< 7 h/nuit"], horizontal=True, key="rsl")
st.session_state.sleep = 1 if "≥ 7" in sleep_r else 0

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# COLONNE CENTRE
# ════════════════════════════════════
f = st.session_state.fruits
s = st.session_state.sport
pct = get_prob(f, s)

if pct >= 65:
    cls="green"; ring_col="#10B981"; chip_cls="chip-green"
    verdict_court = "Profil favorable"
    verdict_long  = f"Les habitudes de vie de {prenom} sont globalement satisfaisantes. Suivi standard recommandé."
elif pct >= 45:
    cls="amber"; ring_col="#F59E0B"; chip_cls="chip-amber"
    verdict_court = "Vigilance recommandée"
    verdict_long  = f"Certaines habitudes de {prenom} présentent des facteurs de risque. Un suivi préventif est conseillé."
else:
    cls="red"; ring_col="#EF4444"; chip_cls="chip-red"
    verdict_court = "Bilan clinique requis"
    verdict_long  = f"Le profil de {prenom} présente des facteurs défavorables. Un bilan approfondi est recommandé."

# Score ring SVG
circ = 2 * 3.14159 * 58
dash_fill = round(circ * pct / 100, 1)
dash_empty = round(circ - dash_fill, 1)

st.markdown('<div class="center-content">', unsafe_allow_html=True)

# Scan + score
st.markdown(f"""
<div class="glass scan-card">
  <svg class="scan-rings" viewBox="0 0 280 280" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="140" cy="140" r="130" stroke="rgba(79,126,255,0.06)" stroke-width="1"/>
    <circle cx="140" cy="140" r="110" stroke="rgba(79,126,255,0.08)" stroke-width="1"/>
    <circle cx="140" cy="140" r="88"  stroke="rgba(79,126,255,0.10)" stroke-width="1"/>
    <circle cx="140" cy="140" r="66"  stroke="rgba(79,126,255,0.12)" stroke-width="1.5"/>
    <line x1="10"  y1="140" x2="270" y2="140" stroke="rgba(79,126,255,0.05)" stroke-width="1"/>
    <line x1="140" y1="10"  x2="140" y2="270" stroke="rgba(79,126,255,0.05)" stroke-width="1"/>
  </svg>
  <div class="scan-figure">👤</div>
  <div class="scan-label">{prenom} {nom} · Analyse en cours</div>

  <div style="margin-top:24px;position:relative;z-index:2;width:100%;">
    <div class="score-ring-wrap">
      <svg class="score-ring-svg" viewBox="0 0 160 160">
        <circle cx="80" cy="80" r="58" fill="none" stroke="rgba(79,126,255,0.1)" stroke-width="10"/>
        <circle cx="80" cy="80" r="58" fill="none" stroke="{ring_col}" stroke-width="10"
          stroke-linecap="round"
          stroke-dasharray="{dash_fill} {dash_empty}"
          stroke-dashoffset="0"/>
      </svg>
      <div class="score-ring-inner">
        <div class="score-num" style="color:{ring_col};">{pct}<span style="font-size:16px">%</span></div>
        <div class="score-unit">Bonne santé</div>
      </div>
    </div>
    <div class="verdict-chip">
      <span class="chip-inner {chip_cls}">{verdict_court}</span>
    </div>
    <div class="verdict-text">{verdict_long}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats chips
st.markdown(f"""
<div class="stat-chips">
  <div class="glass stat-chip chip-1">
    <div class="stat-chip-val">{7 - [f,s,st.session_state.water,st.session_state.sleep,st.session_state.cooking,st.session_state.snacks,st.session_state.skip].count(0)}/7</div>
    <div class="stat-chip-lab">Critères favorables</div>
  </div>
  <div class="glass stat-chip chip-2">
    <div class="stat-chip-val">{[f,s,st.session_state.water,st.session_state.sleep,st.session_state.cooking,st.session_state.snacks,st.session_state.skip].count(0)}</div>
    <div class="stat-chip-lab">Points d'attention</div>
  </div>
  <div class="glass stat-chip chip-3">
    <div class="stat-chip-val">{age}</div>
    <div class="stat-chip-lab">Âge du patient</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Indicateurs
habits = [
    ("Alimentation",   f,                          "favorable" if f else "défavorable"),
    ("Sport",          s,                          "favorable" if s else "défavorable"),
    ("Hydratation",    st.session_state.water,     "favorable" if st.session_state.water else "à améliorer"),
    ("Sommeil",        st.session_state.sleep,     "favorable" if st.session_state.sleep else "insuffisant"),
    ("Cuisine maison", st.session_state.cooking,   "favorable" if st.session_state.cooking else "à revoir"),
    ("Régularité",     st.session_state.skip,      "favorable" if st.session_state.skip else "irrégulier"),
    ("Grignotage",     st.session_state.snacks,    "absent" if st.session_state.snacks else "présent"),
]

st.markdown('<div class="glass" style="padding:18px 22px;">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Indicateurs de mode de vie</div></div>', unsafe_allow_html=True)
for label, val, txt in habits:
    col = "g" if val == 1 else "r"
    st.markdown(f"""
    <div class="ind-item">
      <div class="ind-label">{label}</div>
      <div class="ind-bar-wrap"><div class="ind-bar {col}" style="width:{100 if val else 25}%"></div></div>
      <div class="ind-val {col}">{"✓" if val else "✗"}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# COLONNE DROITE — RECOMMANDATIONS
# ════════════════════════════════════
st.markdown('<div style="display:flex;flex-direction:column;gap:16px;">', unsafe_allow_html=True)
st.markdown('<div class="glass reco-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><div class="section-dot"></div><div class="section-title-text">Recommandations</div></div>', unsafe_allow_html=True)

recos = []
if f == 0:
    recos.append("Augmenter les fruits & légumes à 2 portions minimum par jour.")
if s == 0:
    recos.append("Prescrire 3 séances d'activité physique de 30 min par semaine.")
if st.session_state.snacks == 0:
    recos.append("Réduire le grignotage — favoriser des collations saines si nécessaire.")
if st.session_state.water == 0:
    recos.append("Augmenter l'apport hydrique à 1,5 L d'eau par jour.")
if st.session_state.sleep == 0:
    recos.append("Améliorer l'hygiène du sommeil — objectif : 7 heures par nuit.")
if st.session_state.cooking == 0:
    recos.append("Recommander la cuisine à domicile pour limiter les apports transformés.")
if st.session_state.skip == 0:
    recos.append("Structurer les prises alimentaires en 3 repas réguliers par jour.")

if recos:
    for r in recos:
        st.markdown(f'<div class="reco-item"><div class="reco-bullet"></div><div class="reco-txt">{r}</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="reco-item"><div class="reco-bullet ok"></div><div class="reco-txt">Toutes les habitudes évaluées sont favorables. Encourager le maintien du mode de vie actuel.</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Prochain rendez-vous
st.markdown(f"""
<div class="glass" style="padding:18px 22px;">
  <div class="section-header">
    <div class="section-dot"></div>
    <div class="section-title-text">Suivi recommandé</div>
  </div>
  <div style="font-size:13px;color:#374882;line-height:1.7;">
    {"Bilan clinique à planifier dans les <strong>4 semaines</strong>." if pct < 45 else
     ("Consultation préventive recommandée dans les <strong>3 mois</strong>." if pct < 65 else
      "Reconduire l'évaluation mode de vie lors du <strong>bilan annuel</strong>.")}
  </div>
  <div style="margin-top:14px;padding:12px;background:rgba(79,126,255,0.06);border-radius:8px;border:1px solid rgba(79,126,255,0.12);">
    <div style="font-size:11px;color:#8FA3CC;margin-bottom:2px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">Score de risque global</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:28px;font-weight:700;color:{ring_col};">{pct}%</div>
    <div style="font-size:11px;color:#8FA3CC;">Probabilité d'être en bonne santé</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass" style="padding:14px 18px;">
  <div class="disclaimer-text">
    Outil d'aide à la décision clinique — ne se substitue pas au jugement du praticien.
    Évaluation basée sur les habitudes de vie déclarées par le patient.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div></div></div>', unsafe_allow_html=True)