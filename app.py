import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HealthAI — Aide à la décision clinique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Rajdhani:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #F0F5FF !important;
}
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header, section[data-testid="stSidebar"] { display: none !important; }

/* ── TOP BAR ── */
.topbar {
    background: rgba(255,255,255,0.96);
    border-bottom: 1px solid #D0DCF0;
    padding: 0 36px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 8px rgba(26,95,212,0.07);
    margin-bottom: 0;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, #1A5FD4, #2E7DFF);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-family: 'Rajdhani', sans-serif;
    font-size: 17px; font-weight: 700;
    box-shadow: 0 2px 8px rgba(26,95,212,0.28);
}
.brand-name {
    font-family: 'Rajdhani', sans-serif; font-size: 21px; font-weight: 700;
    color: #1A5FD4; letter-spacing: 0.09em;
}
.brand-sep { width: 1px; height: 22px; background: #D0DCF0; margin: 0 14px; }
.brand-sub { font-size: 12px; color: #4A5E7A; }
.live-badge {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 600; color: #00875A;
    letter-spacing: 0.07em;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #00875A; box-shadow: 0 0 8px rgba(0,135,90,0.5);
    animation: blink 2s ease infinite; display: inline-block;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }
.dr-pill {
    padding: 5px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.07em;
    background: #EBF2FF; color: #1A5FD4;
    border: 1px solid rgba(46,125,255,0.22);
}

/* ── SHELL ── */
.shell {
    display: grid;
    grid-template-columns: 300px 1fr 300px;
    min-height: calc(100vh - 56px);
    background: linear-gradient(145deg, #EEF4FF 0%, #E8F0FF 50%, #EEF8FF 100%);
}

/* ── SIDE PANELS ── */
.side {
    background: rgba(255,255,255,0.88);
    border-right: 1px solid #D0DCF0;
    padding: 20px 18px;
    display: flex; flex-direction: column; gap: 14px;
}
.side.right { border-right: none; border-left: 1px solid #D0DCF0; }

/* ── BLOCKS ── */
.blk {
    background: #fff;
    border: 1px solid #D0DCF0;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(26,95,212,0.04);
}
.blk-hdr {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 12px; padding-bottom: 10px;
    border-bottom: 1px solid #EEF2F8;
}
.blk-line {
    width: 3px; height: 14px; border-radius: 2px;
    background: linear-gradient(180deg, #1A5FD4, #00808C); flex-shrink: 0;
}
.blk-title {
    font-family: 'Rajdhani', sans-serif; font-size: 11px; font-weight: 700;
    color: #4A5E7A; letter-spacing: 0.12em; text-transform: uppercase;
}

/* ── PATIENT ── */
.avatar-zone { text-align: center; margin: 8px 0 14px; }
.avatar-outer {
    width: 72px; height: 72px; border-radius: 50%;
    border: 2px solid #2E7DFF;
    box-shadow: 0 0 0 6px rgba(46,125,255,0.08);
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #EBF2FF, #F0F7FF);
    font-family: 'Rajdhani', sans-serif; font-size: 26px; font-weight: 700; color: #1A5FD4;
}
.p-name {
    font-family: 'Rajdhani', sans-serif; font-size: 17px; font-weight: 700;
    color: #1A2744; margin-bottom: 2px;
}
.p-sub { font-size: 11px; color: #8A9BB8; margin-bottom: 12px; }
.p-div { height: 1px; background: linear-gradient(90deg,transparent,#D0DCF0,transparent); margin-bottom: 10px; }
.info-r { display: flex; justify-content: space-between; padding: 5px 0; font-size: 11.5px; border-bottom: 1px solid #EEF2F8; }
.info-r:last-child { border-bottom: none; }
.ik { color: #8A9BB8; font-weight: 500; }
.iv { color: #1A2744; font-weight: 600; text-align: right; }

/* ── QUESTIONS ── */
.q-r { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid #EEF2F8; }
.q-r:last-child { border-bottom: none; }
.q-l { font-size: 12px; color: #1A2744; flex: 1; line-height: 1.3; padding-right: 8px; }
.tog { display: flex; border: 1px solid #D0DCF0; border-radius: 6px; overflow: hidden; flex-shrink: 0; }
.tb {
    padding: 5px 11px; font-size: 11px; font-weight: 500;
    background: transparent; border: none; cursor: pointer;
    color: #8A9BB8; transition: all 0.12s; font-family: 'Inter', sans-serif;
}
.tb + .tb { border-left: 1px solid #D0DCF0; }
.tb.on { background: #1A5FD4; color: #fff; }
.tb.on-r { background: #C0392B; color: #fff; }

/* ── CENTER ── */
.center-wrap { padding: 20px; display: flex; flex-direction: column; align-items: center; gap: 16px; }

/* ── HOLOGRAM ── */
.holo-area {
    position: relative; width: 100%; max-width: 560px;
    height: 360px; flex-shrink: 0;
}
.holo-area svg.bg { position: absolute; inset: 0; width: 100%; height: 100%; }
.figure-wrap {
    position: absolute; left: 50%; top: 44%;
    transform: translate(-50%, -50%);
    width: 110px; height: 230px;
    filter: drop-shadow(0 0 16px rgba(26,95,212,0.3)) drop-shadow(0 0 36px rgba(0,128,140,0.16));
    animation: float 5s ease-in-out infinite;
}
@keyframes float { 0%,100%{transform:translate(-50%,-50%)} 50%{transform:translate(-50%,-52%)} }
.platform {
    position: absolute; left: 50%; bottom: 16%;
    transform: translateX(-50%);
    width: 90px; height: 8px;
    background: radial-gradient(ellipse, rgba(46,125,255,0.3), transparent 70%);
    animation: glow 3s ease-in-out infinite;
}
@keyframes glow { 0%,100%{opacity:.5} 50%{opacity:1} }
.holo-name {
    position: absolute; bottom: 4%; left: 50%; transform: translateX(-50%);
    font-family: 'Rajdhani', sans-serif; font-size: 11px; font-weight: 600;
    color: #2E7DFF; letter-spacing: 0.16em; text-transform: uppercase;
    white-space: nowrap; opacity: 0.8;
}
.dp {
    position: absolute; display: flex; align-items: center;
    gap: 5px; font-size: 10px; font-weight: 600;
    font-family: 'Rajdhani', sans-serif; letter-spacing: 0.05em;
}
.dp-dot { width: 6px; height: 6px; border-radius: 50%; border: 1.5px solid currentColor; background: #fff; flex-shrink: 0; }
.dp-line { height: 1px; background: currentColor; opacity: 0.4; }
.dp-txt { white-space: nowrap; opacity: 0.85; }

/* ── SCORE ── */
.score-zone {
    width: 100%; max-width: 560px;
    background: #fff; border: 1px solid #D0DCF0;
    border-radius: 12px; padding: 20px 24px;
    display: flex; align-items: center; justify-content: center;
    gap: 28px;
    box-shadow: 0 1px 6px rgba(26,95,212,0.05);
}
.ring-area { position: relative; width: 150px; height: 150px; flex-shrink: 0; }
.ring-area svg { transform: rotate(-90deg); }
.ring-inner {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ring-pct { font-family: 'Rajdhani', sans-serif; font-size: 46px; font-weight: 700; line-height: 1; }
.ring-sub { font-size: 10px; color: #8A9BB8; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }
.verdict-zone { flex: 1; max-width: 280px; }
.v-chip {
    display: inline-block; padding: 5px 14px; border-radius: 5px;
    font-family: 'Rajdhani', sans-serif; font-size: 12px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px;
}
.vc-g { background: #E6F6F0; color: #00875A; border: 1px solid rgba(0,135,90,0.2); }
.vc-a { background: #FFF3E6; color: #B85C00; border: 1px solid rgba(184,92,0,0.2); }
.vc-r { background: #FEF0EF; color: #C0392B; border: 1px solid rgba(192,57,43,0.2); }
.v-desc { font-size: 13px; color: #4A5E7A; line-height: 1.65; }

/* ── STATS ── */
.stats-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; width: 100%; max-width: 560px; }
.sc {
    background: #fff; border: 1px solid #D0DCF0; border-radius: 8px;
    padding: 12px; text-align: center;
    box-shadow: 0 1px 4px rgba(26,95,212,0.04);
}
.sc-val { font-family: 'Rajdhani', sans-serif; font-size: 24px; font-weight: 700; line-height: 1; margin-bottom: 3px; }
.sc-lab { font-size: 9.5px; color: #8A9BB8; letter-spacing: 0.07em; text-transform: uppercase; }

/* ── INDICATORS ── */
.ind-blk { width: 100%; max-width: 560px; background: #fff; border: 1px solid #D0DCF0; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(26,95,212,0.04); }
.ind-r { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #EEF2F8; }
.ind-r:last-child { border-bottom: none; }
.ind-l { font-size: 11.5px; color: #4A5E7A; width: 118px; flex-shrink: 0; }
.ind-t { flex: 1; height: 5px; background: #EEF2F8; border-radius: 99px; overflow: hidden; }
.ind-f { height: 100%; border-radius: 99px; transition: width .45s ease; }
.fg { background: linear-gradient(90deg, #00A86B, #00875A); }
.fr { background: linear-gradient(90deg, #E08080, #C0392B); }
.ind-s { font-size: 10.5px; font-weight: 600; width: 78px; text-align: right; flex-shrink: 0; }

/* ── RECO ── */
.reco-item { display: flex; gap: 9px; padding: 8px 0; border-bottom: 1px solid #EEF2F8; align-items: flex-start; }
.reco-item:last-child { border-bottom: none; }
.reco-bar { width: 3px; border-radius: 2px; align-self: stretch; flex-shrink: 0; min-height: 18px; }
.rb-b { background: #2E7DFF; }
.rb-g { background: #00875A; }
.reco-txt { font-size: 12.5px; color: #1A2744; line-height: 1.6; }

.alert-b { border-radius: 8px; padding: 11px 14px; font-size: 12.5px; color: #1A2744; line-height: 1.6; margin-top: 8px; }
.ab-g { background: #E6F6F0; border: 1px solid rgba(0,135,90,0.2); }
.ab-a { background: #FFF3E6; border: 1px solid rgba(184,92,0,0.2); }
.ab-r { background: #FEF0EF; border: 1px solid rgba(192,57,43,0.2); }

.fu-box { margin-top: 10px; padding: 14px; background: #EBF2FF; border-radius: 8px; border: 1px solid rgba(46,125,255,0.15); text-align: center; }
.fu-val { font-family: 'Rajdhani', sans-serif; font-size: 32px; font-weight: 700; line-height: 1; }
.fu-lbl { font-size: 10px; color: #8A9BB8; margin-top: 3px; letter-spacing: 0.06em; text-transform: uppercase; }

.disc { font-size: 10.5px; color: #8A9BB8; padding: 10px 14px; text-align: center; background: #F8FAFF; border: 1px solid #D0DCF0; border-radius: 8px; line-height: 1.5; }

/* ── CONTEXT BANNER ── */
.context-banner {
    background: #EBF2FF; border: 1px solid rgba(46,125,255,0.2);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 4px;
    font-size: 12px; color: #1A5FD4; line-height: 1.6;
}
.context-banner strong { color: #1A2744; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ──
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
    for c,m in enc.items():
        df[c+'_enc'] = df[c].map(m)
    df['y'] = (df['healthy']=='Oui').astype(int)
    X = df[[c+'_enc' for c in enc]].values
    y = df['y'].values
    mod = LogisticRegression(penalty='l1',C=0.5,class_weight={0:5,1:1},
                             solver='liblinear',max_iter=2000,random_state=42)
    mod.fit(X, y)
    loo = LeaveOneOut()
    cv_p = cross_val_predict(mod, X, y, cv=loo, method='predict_proba')[:,1]
    return mod, X, y, cv_p

model, X, y, cv_probs = load_model()

def sigmoid(z): return 1/(1+np.exp(-z))
def get_prob(f,s): return round(sigmoid(-0.8624+1.881*f+0.740*s)*100,1)

# ── TOP BAR ──
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">H</div>
    <span class="brand-name">HEALTHAI</span>
    <div class="brand-sep"></div>
    <span class="brand-sub">Système d'aide à la décision clinique · Mode de vie & Prévention</span>
  </div>
  <div style="display:flex;align-items:center;gap:14px;">
    <span class="live-badge"><span class="live-dot"></span>&nbsp;SYSTÈME ACTIF</span>
    <span class="dr-pill">INTERFACE MÉDECIN</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="shell">', unsafe_allow_html=True)

# ══ SIDE LEFT ══
st.markdown('<div class="side">', unsafe_allow_html=True)

# Context banner
st.markdown("""
<div class="context-banner">
  <strong>À propos de cet outil</strong><br>
  HealthAI analyse les habitudes de vie déclarées d'un patient pour estimer sa probabilité d'être en bonne santé.
  Renseignez les informations du patient, puis ajustez ses habitudes pour obtenir une évaluation instantanée.
</div>
""", unsafe_allow_html=True)

# Patient identity
with st.form("patient_form"):
    st.markdown('<div class="blk"><div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">Dossier patient</div></div>', unsafe_allow_html=True)
    prenom  = st.text_input("Prénom",    value="Marie",      label_visibility="collapsed", placeholder="Prénom")
    nom     = st.text_input("Nom",       value="Dupont",     label_visibility="collapsed", placeholder="Nom de famille")
    age     = st.number_input("Âge",     value=42, min_value=1, max_value=120, label_visibility="collapsed")
    medecin = st.text_input("Médecin",   value="Dr. Martin", label_visibility="collapsed", placeholder="Médecin traitant")
    st.form_submit_button("Mettre à jour le dossier", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

ini = (prenom[0] if prenom else "?") + (nom[0] if nom else "?")

st.markdown(f"""
<div class="blk">
  <div class="avatar-zone">
    <div class="avatar-outer">{ini.upper()}</div>
  </div>
  <div class="p-name" style="text-align:center">{prenom} {nom}</div>
  <div class="p-sub" style="text-align:center">{age} ans · Consultation mode de vie</div>
  <div class="p-div"></div>
  <div class="info-r"><span class="ik">Médecin traitant</span><span class="iv">{medecin}</span></div>
  <div class="info-r"><span class="ik">Type d'évaluation</span><span class="iv">Mode de vie</span></div>
  <div class="info-r"><span class="ik">Outil clinique</span><span class="iv">HealthAI v1.0</span></div>
  <div class="info-r"><span class="ik">Modèle</span><span class="iv">Validé · n = 19</span></div>
</div>
""", unsafe_allow_html=True)

# Questions alimentation
st.markdown('<div class="blk"><div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">Alimentation</div></div>', unsafe_allow_html=True)
fruits  = st.radio("Fruits & légumes / jour",   ["2 portions ou plus","Moins de 2 portions"],   horizontal=True, key="fruits")
cooking = st.radio("Mode alimentaire",           ["Cuisine à domicile","Plats préparés"],         horizontal=True, key="cooking")
snacks  = st.radio("Grignotage inter-repas",     ["Absent ou rare","Régulier"],                   horizontal=True, key="snacks")
skip    = st.radio("Régularité des repas",       ["Repas réguliers","Repas sautés"],              horizontal=True, key="skip")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="blk"><div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">Hydratation & Sommeil</div></div>', unsafe_allow_html=True)
water = st.radio("Consommation d'eau / jour",   ["1,5 L ou plus","Moins de 1,5 L"], horizontal=True, key="water")
sleep = st.radio("Durée du sommeil / nuit",     ["7 h ou plus","Moins de 7 h"],      horizontal=True, key="sleep")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="blk"><div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">Activité physique</div></div>', unsafe_allow_html=True)
sport = st.radio("Séances / semaine (≥ 30 min)", ["3 séances ou plus","Moins de 3 séances"], horizontal=True, key="sport")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end side left

# ── Valeurs encodées ──
fv = 1 if "2 portions" in fruits else 0
sv = 1 if "3 séances"  in sport  else 0
wv = 1 if "1,5 L ou"  in water  else 0
slv= 1 if "7 h ou"    in sleep  else 0
cv = 1 if "domicile"  in cooking else 0
snv= 1 if "Absent"    in snacks  else 0
skv= 1 if "réguliers" in skip   else 0

pct  = get_prob(fv, sv)
vals = [fv, sv, wv, slv, cv, snv, skv]
good = sum(vals)
bad  = 7 - good

# Verdict
if pct >= 65:
    col="color:#00875A"; ring_col="#00875A"
    chip="vc-g"; chip_txt="Profil favorable"
    v_desc=f"Les habitudes de vie de {prenom} sont globalement satisfaisantes. Un suivi standard est recommandé. Encourager le maintien du mode de vie actuel."
    fu_txt="Reconduire l'évaluation lors du bilan annuel."
    alert_cls="ab-g"
    alert_txt=f"Aucune alerte clinique. Tous les critères majeurs sont dans les normes recommandées."
elif pct >= 45:
    col="color:#B85C00"; ring_col="#B85C00"
    chip="vc-a"; chip_txt="Vigilance recommandée"
    v_desc=f"Certaines habitudes de {prenom} présentent des facteurs de risque modérés. Un suivi préventif renforcé est conseillé."
    fu_txt="Consultation préventive recommandée dans les 3 mois."
    alert_cls="ab-a"
    alert_txt=f"Des ajustements comportementaux ciblés pourraient améliorer significativement le profil de santé."
else:
    col="color:#C0392B"; ring_col="#C0392B"
    chip="vc-r"; chip_txt="Bilan clinique requis"
    v_desc=f"Le profil de {prenom} présente plusieurs facteurs défavorables cumulés. Un bilan clinique approfondi est recommandé sans délai."
    fu_txt="Bilan clinique à planifier dans les 4 semaines."
    alert_cls="ab-r"
    alert_txt=f"Plusieurs facteurs de risque identifiés. Ne pas différer la prise en charge."

# Ring SVG
C = 2*3.14159*60
fill = C*pct/100
ring_svg = f"""
<svg width="150" height="150" viewBox="0 0 150 150">
  <circle cx="75" cy="75" r="60" fill="none" stroke="rgba(26,95,212,0.1)" stroke-width="9"/>
  <circle cx="75" cy="75" r="60" fill="none" stroke="{ring_col}" stroke-width="9"
    stroke-linecap="round"
    stroke-dasharray="{round(fill,2)} {round(C-fill,2)}"
    stroke-dashoffset="0"/>
</svg>
"""

# ══ CENTER ══
st.markdown('<div style="background:linear-gradient(145deg,#EEF4FF,#E8F0FF,#EEF8FF);display:flex;flex-direction:column;align-items:center;padding:20px;gap:16px;">', unsafe_allow_html=True)

# Hologram
st.markdown(f"""
<div class="holo-area">
  <svg class="bg" viewBox="0 0 560 360" fill="none" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="280" cy="195" rx="250" ry="180" stroke="rgba(46,125,255,0.05)" stroke-width="1"/>
    <ellipse cx="280" cy="195" rx="210" ry="150" stroke="rgba(46,125,255,0.07)" stroke-width="1"/>
    <ellipse cx="280" cy="195" rx="165" ry="118" stroke="rgba(46,125,255,0.09)" stroke-width="1"/>
    <ellipse cx="280" cy="195" rx="118" ry="84"  stroke="rgba(46,125,255,0.08)" stroke-width="1.5"/>
    <ellipse cx="280" cy="195" rx="70"  ry="50"  stroke="rgba(46,125,255,0.05)" stroke-width="1"/>
    <line x1="20"  y1="195" x2="540" y2="195" stroke="rgba(46,125,255,0.05)" stroke-width="1"/>
    <line x1="280" y1="15"  x2="280" y2="350" stroke="rgba(46,125,255,0.05)" stroke-width="1"/>
    <line x1="20"  y1="110" x2="540" y2="110" stroke="rgba(46,125,255,0.03)" stroke-width=".5" stroke-dasharray="4 8"/>
    <line x1="20"  y1="290" x2="540" y2="290" stroke="rgba(46,125,255,0.03)" stroke-width=".5" stroke-dasharray="4 8"/>
    <path d="M22 28 L22 18 L32 18" stroke="rgba(46,125,255,0.28)" stroke-width="1.5" fill="none"/>
    <path d="M528 18 L538 18 L538 28" stroke="rgba(46,125,255,0.28)" stroke-width="1.5" fill="none"/>
    <path d="M22 342 L22 352 L32 352" stroke="rgba(46,125,255,0.28)" stroke-width="1.5" fill="none"/>
    <path d="M538 342 L538 352 L528 352" stroke="rgba(46,125,255,0.28)" stroke-width="1.5" fill="none"/>
    <line x1="110" y1="100" x2="218" y2="138" stroke="rgba(46,125,255,0.18)" stroke-width="1"/>
    <line x1="80"  y1="195" x2="200" y2="195" stroke="rgba(0,128,140,0.18)" stroke-width="1"/>
    <line x1="100" y1="285" x2="210" y2="248" stroke="rgba(46,125,255,0.18)" stroke-width="1"/>
    <line x1="450" y1="100" x2="342" y2="138" stroke="rgba(46,125,255,0.18)" stroke-width="1"/>
    <line x1="480" y1="195" x2="360" y2="195" stroke="rgba(0,128,140,0.18)" stroke-width="1"/>
    <line x1="460" y1="285" x2="350" y2="248" stroke="rgba(46,125,255,0.18)" stroke-width="1"/>
  </svg>

  <svg class="figure-wrap" viewBox="0 0 80 200" fill="none" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="40" cy="17" rx="13" ry="14" stroke="#1A5FD4" stroke-width="1.4" fill="rgba(46,125,255,0.07)"/>
    <rect x="35.5" y="30" width="9" height="8" rx="2" stroke="#1A5FD4" stroke-width="1.2" fill="rgba(46,125,255,0.05)"/>
    <path d="M22 38 Q17 42 17 58 L17 98 Q17 102 21 102 L59 102 Q63 102 63 98 L63 58 Q63 42 58 38 Z"
      stroke="#1A5FD4" stroke-width="1.4" fill="rgba(46,125,255,0.06)"/>
    <path d="M22 42 Q11 46 9 66 L7 88 Q6 92 9 93 L13 93 Q16 92 17 88 L19 66 Z"
      stroke="#1A5FD4" stroke-width="1.2" fill="rgba(46,125,255,0.05)"/>
    <path d="M58 42 Q69 46 71 66 L73 88 Q74 92 71 93 L67 93 Q64 92 63 88 L61 66 Z"
      stroke="#1A5FD4" stroke-width="1.2" fill="rgba(46,125,255,0.05)"/>
    <path d="M23 102 L21 138 L19 168 Q18 173 21 174 L29 174 Q32 173 33 168 L35 138 L37 102 Z"
      stroke="#1A5FD4" stroke-width="1.2" fill="rgba(46,125,255,0.05)"/>
    <path d="M57 102 L59 138 L61 168 Q62 173 59 174 L51 174 Q48 173 47 168 L45 138 L43 102 Z"
      stroke="#1A5FD4" stroke-width="1.2" fill="rgba(46,125,255,0.05)"/>
    <ellipse cx="25" cy="180" rx="8" ry="5" stroke="#1A5FD4" stroke-width="1" fill="rgba(46,125,255,0.05)"/>
    <ellipse cx="55" cy="180" rx="8" ry="5" stroke="#1A5FD4" stroke-width="1" fill="rgba(46,125,255,0.05)"/>
    <circle cx="34" cy="60" r="4" fill="none" stroke="{'#C0392B' if pct < 45 else '#00875A'}" stroke-width="1.2"/>
    <circle cx="34" cy="60" r="1.5" fill="{'#C0392B' if pct < 45 else '#00875A'}" opacity="0.6"/>
    <line x1="40" y1="38" x2="40" y2="100" stroke="rgba(46,125,255,0.25)" stroke-width=".8" stroke-dasharray="3 3"/>
    <line x1="24" y1="68" x2="56" y2="68" stroke="rgba(46,125,255,0.14)" stroke-width=".8"/>
    <line x1="22" y1="82" x2="58" y2="82" stroke="rgba(46,125,255,0.14)" stroke-width=".8"/>
  </svg>

  <div class="platform"></div>

  <div class="dp" style="left:8px;top:20%;color:#1A5FD4;flex-direction:row;">
    <div class="dp-dot"></div><div class="dp-line" style="width:22px"></div>
    <div class="dp-txt">Alimentation — {'OK' if fv else 'Alerte'}</div>
  </div>
  <div class="dp" style="left:4px;top:46%;color:{'#00808C' if wv else '#C0392B'};flex-direction:row;">
    <div class="dp-dot"></div><div class="dp-line" style="width:22px"></div>
    <div class="dp-txt">Hydratation — {'OK' if wv else 'Insuffisante'}</div>
  </div>
  <div class="dp" style="left:6px;top:70%;color:{'#1A5FD4' if sv else '#C0392B'};flex-direction:row;">
    <div class="dp-dot"></div><div class="dp-line" style="width:22px"></div>
    <div class="dp-txt">Activité — {'OK' if sv else 'Insuffisante'}</div>
  </div>
  <div class="dp" style="right:6px;top:20%;color:{'#1A5FD4' if slv else '#C0392B'};flex-direction:row-reverse;">
    <div class="dp-dot"></div><div class="dp-line" style="width:22px"></div>
    <div class="dp-txt">Sommeil — {'OK' if slv else 'Insuffisant'}</div>
  </div>
  <div class="dp" style="right:2px;top:46%;color:{'#00808C' if cv else '#C0392B'};flex-direction:row-reverse;">
    <div class="dp-dot"></div><div class="dp-line" style="width:22px"></div>
    <div class="dp-txt">Cuisine — {'OK' if cv else 'À revoir'}</div>
  </div>
  <div class="dp" style="right:4px;top:70%;color:{'#1A5FD4' if snv else '#C0392B'};flex-direction:row-reverse;">
    <div class="dp-dot"></div><div class="dp-line" style="width:22px"></div>
    <div class="dp-txt">Grignotage — {'Absent' if snv else 'Présent'}</div>
  </div>

  <div class="holo-name">{prenom} {nom} · Analyse mode de vie</div>
</div>
""", unsafe_allow_html=True)

# Score centré
st.markdown(f"""
<div class="score-zone">
  <div class="ring-area">
    <svg width="150" height="150" viewBox="0 0 150 150">
      <circle cx="75" cy="75" r="60" fill="none" stroke="rgba(26,95,212,0.1)" stroke-width="9"/>
      <circle cx="75" cy="75" r="60" fill="none" stroke="{ring_col}" stroke-width="9"
        stroke-linecap="round"
        stroke-dasharray="{round(fill,1)} {round(C-fill,1)}"
        stroke-dashoffset="0"/>
    </svg>
    <div class="ring-inner">
      <div class="ring-pct" style="{col}">{pct}%</div>
      <div class="ring-sub">Score santé</div>
    </div>
  </div>
  <div class="verdict-zone">
    <div class="v-chip {chip}">{chip_txt}</div>
    <div class="v-desc">{v_desc}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown(f"""
<div class="stats-row">
  <div class="sc">
    <div class="sc-val" style="{col}">{good}/7</div>
    <div class="sc-lab">Critères favorables</div>
  </div>
  <div class="sc">
    <div class="sc-val" style="{'color:#C0392B' if bad>0 else 'color:#00875A'}">{bad}</div>
    <div class="sc-lab">Points d'attention</div>
  </div>
  <div class="sc">
    <div class="sc-val" style="color:#1A5FD4">{age}</div>
    <div class="sc-lab">Âge du patient</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Indicators
ind_cfg = [
    ("Fruits & légumes", fv,  "Favorable","Insuffisant"),
    ("Activité physique", sv, "Favorable","Insuffisante"),
    ("Hydratation",       wv, "Favorable","Insuffisante"),
    ("Sommeil",           slv,"Favorable","Insuffisant"),
    ("Cuisine maison",    cv, "Favorable","À revoir"),
    ("Régularité repas",  skv,"Réguliers","Irréguliers"),
    ("Grignotage",        snv,"Absent",   "Présent"),
]
rows = ""
for lbl, v, ok_txt, ko_txt in ind_cfg:
    bar_cls = "fg" if v else "fr"
    bar_w   = "100%" if v else "16%"
    sc      = "color:#00875A" if v else "color:#C0392B"
    txt     = ok_txt if v else ko_txt
    rows += f'<div class="ind-r"><div class="ind-l">{lbl}</div><div class="ind-t"><div class="ind-f {bar_cls}" style="width:{bar_w}"></div></div><div class="ind-s" style="{sc}">{txt}</div></div>'

st.markdown(f"""
<div class="ind-blk">
  <div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">Indicateurs de mode de vie</div></div>
  {rows}
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end center

# ══ SIDE RIGHT ══
st.markdown('<div class="side right">', unsafe_allow_html=True)

# Recommandations
recos = []
if not fv:  recos.append("Augmenter la consommation de fruits et légumes à 2 portions minimum par jour.")
if not sv:  recos.append("Prescrire 3 séances d'activité physique de 30 minutes par semaine.")
if not snv: recos.append("Réduire le grignotage — favoriser des collations saines si nécessaire.")
if not wv:  recos.append("Augmenter l'apport hydrique à 1,5 litre d'eau par jour minimum.")
if not slv: recos.append("Améliorer l'hygiène du sommeil — objectif : 7 heures par nuit.")
if not cv:  recos.append("Recommander la cuisine à domicile pour limiter les apports transformés.")
if not skv: recos.append("Structurer les prises alimentaires en 3 repas réguliers par jour.")

reco_html = ""
if recos:
    for r in recos:
        reco_html += f'<div class="reco-item"><div class="reco-bar rb-b"></div><div class="reco-txt">{r}</div></div>'
else:
    reco_html = '<div class="reco-item"><div class="reco-bar rb-g"></div><div class="reco-txt">Toutes les habitudes évaluées sont favorables. Encourager le maintien du mode de vie actuel.</div></div>'

st.markdown(f"""
<div class="blk">
  <div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">Recommandations cliniques</div></div>
  {reco_html}
</div>
""", unsafe_allow_html=True)

# Suivi
st.markdown(f"""
<div class="blk">
  <div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">Suivi recommandé</div></div>
  <div style="font-size:12.5px;color:#4A5E7A;line-height:1.65">{fu_txt}</div>
  <div class="fu-box">
    <div class="fu-val" style="{col}">{pct}%</div>
    <div class="fu-lbl">Score de santé</div>
  </div>
  <div class="alert-b {alert_cls}">{alert_txt}</div>
</div>
""", unsafe_allow_html=True)

# À propos du modèle
st.markdown(f"""
<div class="blk">
  <div class="blk-hdr"><div class="blk-line"></div><div class="blk-title">À propos du modèle</div></div>
  <div style="font-size:12px;color:#4A5E7A;line-height:1.7;">
    Ce score est calculé par un modèle de classification supervisée entraîné sur des données
    collectées via sondage (n = 19 patients). Le modèle a été optimisé pour ne manquer aucun
    profil à risque — il peut générer des alertes conservatrices sur des profils intermédiaires.<br><br>
    <strong style="color:#1A2744;">Sondage :</strong> 8 questions sur les habitudes de vie (alimentation, sport, sommeil, hydratation).<br>
    <strong style="color:#1A2744;">Précision :</strong> 84.2% · Rappel sur profils à risque : 100%.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disc">
  Outil d'aide à la décision clinique.<br>
  Ne se substitue pas au jugement du praticien ni à un examen clinique complet.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end side right
st.markdown('</div>', unsafe_allow_html=True)  # end shell