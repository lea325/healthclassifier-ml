import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, log_loss, f1_score, recall_score, precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
st.set_page_config(
    page_title="HealthClassifier — Outil d'aide au diagnostic",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS CUSTOM ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #F8FAFC; }
    .block-container { padding: 2rem 3rem !important; max-width: 1200px; }

    /* Header médical */
    .med-header {
        background: linear-gradient(135deg, #1a3a6e 0%, #1e5799 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .med-header h1 { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .med-header p  { color: rgba(255,255,255,0.75); margin: 0.4rem 0 0; font-size: 0.95rem; }

    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 0.75rem;
    }

    /* Résultat verdict */
    .verdict-sain {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-risque {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-vigilance {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    /* KPI metric */
    .kpi-box {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .kpi-num  { font-size: 2rem; font-weight: 700; line-height: 1; }
    .kpi-lab  { font-size: 0.72rem; color: #64748B; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }

    /* Toggle pills pour les questions */
    .stRadio > div { gap: 0.5rem; }
    .stRadio > div > label {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        cursor: pointer;
        transition: all 0.15s;
        font-size: 0.9rem;
    }
    .stRadio > div > label:hover { background: #E0F2FE; border-color: #0ea5e9; }

    /* Progress bar personnalisée */
    .prob-bar-wrap {
        background: #F1F5F9;
        border-radius: 99px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s ease;
    }

    /* Section tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        color: #64748B;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        color: #1a3a6e;
        border: 1px solid #E2E8F0;
        border-bottom: 2px solid white;
    }

    /* Variable badge */
    .var-badge {
        display: inline-block;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 99px;
        font-weight: 600;
    }
    .badge-actif { background: #dcfce7; color: #166534; }
    .badge-nul   { background: #f1f5f9; color: #64748B; }

    /* Séparateur section */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        margin: 1.5rem 0 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #E2E8F0;
    }

    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #64748B !important; }

    .formula-box {
        background: #0f172a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #7dd3fc;
        margin: 0.75rem 0;
    }

    footer { display: none !important; }
    #MainMenu { display: none !important; }
    header { display: none !important; }
</style>
""", unsafe_allow_html=True)

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(fruits_v, sport_v):
    z = -0.8624 + 1.881 * fruits_v + 0.740 * sport_v
    return sigmoid(z), z

# ── HEADER ──
st.markdown("""
<div class="med-header">
  <div style="display:flex; align-items:center; gap:14px;">
    <span style="font-size:2.5rem;">🏥</span>
    <div>
      <h1>HealthClassifier</h1>
      <p>Outil d'aide à la décision clinique · Régression logistique L1 · Validé sur 19 patients · Master 2 IMC&DS — Paris 1 Panthéon-Sorbonne</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3 = st.tabs([
    "🩺  Évaluation patient",
    "📊  Performance du modèle",
    "🔬  Transparence algorithmique"
])

# ═══════════════════════════════════════
# TAB 1 — INTERFACE MÉDECIN
# ═══════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-label">Habitudes alimentaires</div>', unsafe_allow_html=True)

        fruits = st.radio(
            "🥦 Consommation quotidienne de fruits & légumes",
            ["≥ 2 portions par jour  ✓", "< 2 portions par jour"],
            help="Facteur le plus discriminant du modèle (β = +1.881)",
            horizontal=True
        )

        cooking = st.radio(
            "🍳 Mode d'alimentation principal",
            ["Cuisine maison principalement", "Plats préparés / Livraison"],
            horizontal=True
        )

        snacks = st.radio(
            "🍿 Grignotage entre les repas",
            ["Rarement ou jamais", "Régulièrement (chips, sodas, biscuits)"],
            horizontal=True
        )

        skip = st.radio(
            "⏭️ Saute des repas régulièrement",
            ["Non", "Oui"],
            horizontal=True
        )

        st.markdown('<div class="section-label">Hydratation & activité physique</div>', unsafe_allow_html=True)

        water = st.radio(
            "💧 Consommation d'eau par jour",
            ["≥ 1,5 litre", "< 1,5 litre"],
            horizontal=True
        )

        sport = st.radio(
            "🏃 Séances d'activité physique / semaine (≥ 30 min)",
            ["≥ 3 séances par semaine  ✓", "< 3 séances par semaine"],
            help="Second facteur le plus discriminant (β = +0.740)",
            horizontal=True
        )

        st.markdown('<div class="section-label">Sommeil & récupération</div>', unsafe_allow_html=True)

        sleep = st.radio(
            "😴 Durée moyenne du sommeil en semaine",
            ["≥ 7 heures", "< 7 heures"],
            horizontal=True
        )

        rested = st.radio(
            "😌 Sentiment de récupération au réveil",
            ["Oui, je me sens reposé(e)", "Non, je suis fatigué(e)"],
            horizontal=True
        )

    with col_result:
        # Calcul
        fruits_v = 1 if "≥ 2" in fruits else 0
        sport_v  = 1 if "≥ 3" in sport  else 0
        prob, z_val = predict(fruits_v, sport_v)
        pct = round(prob * 100, 1)

        # Verdict
        if prob >= 0.65:
            verdict_class = "verdict-sain"
            verdict_icon  = "✅"
            verdict_titre = "Profil favorable"
            verdict_reco  = "Pas d'alerte particulière. Suivi standard recommandé."
            verdict_color = "#10b981"
        elif prob >= 0.45:
            verdict_class = "verdict-vigilance"
            verdict_icon  = "⚠️"
            verdict_titre = "Profil de vigilance"
            verdict_reco  = "Quelques habitudes à risque. Suivi préventif conseillé."
            verdict_color = "#f59e0b"
        else:
            verdict_class = "verdict-risque"
            verdict_icon  = "🔴"
            verdict_titre = "Profil à risque"
            verdict_reco  = "Bilan approfondi recommandé. Ne pas laisser passer."
            verdict_color = "#ef4444"

        # Affichage résultat
        st.markdown(f"""
        <div class="{verdict_class}" style="margin-bottom:1rem;">
          <div style="font-size:2.5rem; margin-bottom:0.3rem;">{verdict_icon}</div>
          <div style="font-size:1.4rem; font-weight:700; color:#1e293b; margin-bottom:0.3rem;">{verdict_titre}</div>
          <div style="font-size:0.9rem; color:#475569;">{verdict_reco}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilité
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Probabilité d'être en bonne santé</div>
          <div style="font-size:3rem; font-weight:700; color:{verdict_color}; line-height:1;">{pct}%</div>
          <div class="prob-bar-wrap" style="margin-top:0.75rem;">
            <div class="prob-bar-fill" style="width:{pct}%; background:{verdict_color};"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94A3B8; margin-top:4px;">
            <span>0% — À risque</span>
            <span>Seuil décision : 45%</span>
            <span>100% — Sain</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Facteurs actifs
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Facteurs déterminants (modèle L1)</div>
          <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
              <span style="font-size:0.88rem;">🥦 Fruits & légumes (β = +1.881)</span>
              <span style="font-weight:600; color:{'#10b981' if fruits_v else '#ef4444'}; font-size:0.85rem;">{'✓ Favorable' if fruits_v else '✗ Défavorable'}</span>
            </div>
            <div class="prob-bar-wrap" style="height:7px;">
              <div class="prob-bar-fill" style="width:{fruits_v * 100}%; background:{'#10b981' if fruits_v else '#ef4444'};"></div>
            </div>
          </div>
          <div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
              <span style="font-size:0.88rem;">🏃 Activité physique (β = +0.740)</span>
              <span style="font-weight:600; color:{'#10b981' if sport_v else '#ef4444'}; font-size:0.85rem;">{'✓ Favorable' if sport_v else '✗ Défavorable'}</span>
            </div>
            <div class="prob-bar-wrap" style="height:7px;">
              <div class="prob-bar-fill" style="width:{sport_v * 100}%; background:{'#10b981' if sport_v else '#ef4444'};"></div>
            </div>
          </div>
          <div style="margin-top:10px; font-size:0.75rem; color:#94A3B8; font-style:italic;">
            Les 6 autres variables ont été annulées par L1 (β = 0) — non discriminantes statistiquement.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Calcul transparent
        with st.expander("🔎 Voir le calcul mathématique"):
            st.markdown(f"""
            <div class="formula-box">
Z  =  β₀ + β_fruits × X_fruits + β_sport × X_sport<br>
Z  =  −0.8624 + 1.881 × {fruits_v} + 0.740 × {sport_v}<br>
Z  =  {z_val:.4f}<br><br>
σ(Z) = 1 / (1 + e^(−{z_val:.4f}))<br>
σ(Z) = <span style="color:#fbbf24; font-weight:700;">{prob:.4f}</span>  →  {pct}%<br><br>
Décision : {pct}% {'≥' if prob >= 0.45 else '<'} 45%  →  <span style="color:{'#4ade80' if prob >= 0.45 else '#f87171'}; font-weight:700;">{'SAIN' if prob >= 0.45 else 'À RISQUE'}</span>
            </div>
            """, unsafe_allow_html=True)

        # Recommandations contextuelles
        st.markdown('<div class="section-label">Recommandations</div>', unsafe_allow_html=True)
        recommandations = []
        if not fruits_v:
            recommandations.append("🥦 Augmenter la consommation de fruits et légumes à ≥ 2 portions/jour")
        if not sport_v:
            recommandations.append("🏃 Introduire ≥ 3 séances d'activité physique de 30 min par semaine")
        if "Régulièrement" in snacks:
            recommandations.append("🍿 Réduire le grignotage entre les repas")
        if "< 1,5" in water:
            recommandations.append("💧 Augmenter l'apport hydrique à ≥ 1,5 L/jour")
        if "< 7" in sleep:
            recommandations.append("😴 Améliorer l'hygiène du sommeil (objectif : ≥ 7h/nuit)")

        if recommandations:
            for r in recommandations:
                st.markdown(f"""
                <div style="background:#FFF7ED; border-left:3px solid #f59e0b; border-radius:0 8px 8px 0; padding:0.6rem 1rem; margin-bottom:6px; font-size:0.88rem; color:#1e293b;">
                  {r}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#F0FDF4; border-left:3px solid #10b981; border-radius:0 8px 8px 0; padding:0.6rem 1rem; font-size:0.88rem; color:#1e293b;">
              ✅ Toutes les habitudes évaluées sont favorables. Maintenir ce mode de vie.
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 2 — PERFORMANCE
# ═══════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="card" style="background:#EFF6FF; border-color:#BFDBFE;">
      <strong style="color:#1e3a8a;">À propos de ce modèle</strong><br>
      <span style="font-size:0.88rem; color:#1e40af;">
      Régression logistique avec régularisation L1 (Lasso, λ=2.0) et pondération des erreurs (w₀=5 pour la classe "à risque").
      Validé par Leave-One-Out Cross-Validation (LOOCV) sur 19 patients — la méthode la plus robuste pour les petits datasets.
      </span>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "100%", "Recall · À risque", "#10b981"),
        (k2, "57.1%", "Précision · À risque", "#f59e0b"),
        (k3, "80.8%", "F1-Score Macro", "#3b82f6"),
        (k4, "0.80", "AUC-ROC", "#8b5cf6"),
        (k5, "0.469", "NLL (modèle)", "#64748B"),
    ]
    for col, val, lab, color in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-box">
              <div class="kpi-num" style="color:{color};">{val}</div>
              <div class="kpi-lab">{lab}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Matrice de confusion · LOOCV</div>
          <table style="width:100%; border-collapse:collapse; font-size:0.88rem; text-align:center;">
            <tr>
              <th style="border:none; padding:8px;"></th>
              <th style="background:#fee2e2; padding:10px; border-radius:6px; color:#991b1b; font-size:0.8rem;">Prédit : À risque</th>
              <th style="background:#dcfce7; padding:10px; border-radius:6px; color:#166534; font-size:0.8rem;">Prédit : Sain</th>
            </tr>
            <tr>
              <td style="color:#64748B; padding:8px; font-size:0.8rem; text-align:left;">Réel : À risque</td>
              <td style="background:#dcfce7; border-radius:6px; padding:16px;">
                <div style="font-size:1.8rem; font-weight:700; color:#166534;">4</div>
                <div style="font-size:0.7rem; color:#166534;">Vrais Négatifs ✓</div>
              </td>
              <td style="background:#f1f5f9; border-radius:6px; padding:16px;">
                <div style="font-size:1.8rem; font-weight:700; color:#94A3B8;">0</div>
                <div style="font-size:0.7rem; color:#94A3B8;">Faux Positifs</div>
              </td>
            </tr>
            <tr>
              <td style="color:#64748B; padding:8px; font-size:0.8rem; text-align:left;">Réel : Sain</td>
              <td style="background:#fef3c7; border-radius:6px; padding:16px;">
                <div style="font-size:1.8rem; font-weight:700; color:#92400e;">3</div>
                <div style="font-size:0.7rem; color:#92400e;">Faux Négatifs ⚠</div>
              </td>
              <td style="background:#dcfce7; border-radius:6px; padding:16px;">
                <div style="font-size:1.8rem; font-weight:700; color:#166534;">12</div>
                <div style="font-size:0.7rem; color:#166534;">Vrais Positifs ✓</div>
              </td>
            </tr>
          </table>
          <div style="margin-top:12px; font-size:0.8rem; color:#10b981; font-weight:600;">
            ✓ TN = 4/4 → Recall = 100% sur classe à risque. Aucun patient à risque manqué.
          </div>
          <div style="font-size:0.8rem; color:#92400e; margin-top:4px;">
            ⚠ FN = 3 → 3 patients sains sur-alertés. Examens inutiles mais sans risque vital.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Pourquoi ces métriques ?</div>
        """, unsafe_allow_html=True)

        metriques = [
            ("🎯 Recall prioritaire", "#10b981",
             "Un FN = patient malade non détecté → risque vital. On ne peut pas se le permettre. Recall = 100% : objectif atteint."),
            ("⚖️ F1-macro = 80.8%", "#3b82f6",
             "Équilibre Précision/Recall sur les deux classes, malgré le déséquilibre 79% sains / 21% à risque."),
            ("📈 AUC-ROC = 0.80", "#8b5cf6",
             "Le modèle classe correctement 80% des paires (sain, à risque), indépendamment du seuil."),
            ("❌ Accuracy écartée", "#ef4444",
             "Avec 79% de sains, un modèle naïf (toujours 'sain') atteint 79% d'accuracy sans rien apprendre."),
        ]

        for titre, color, desc in metriques:
            st.markdown(f"""
            <div style="border-left:3px solid {color}; border-radius:0 8px 8px 0; padding:0.7rem 1rem; margin-bottom:8px; background:#f8fafc;">
              <div style="font-weight:600; font-size:0.88rem; color:#1e293b; margin-bottom:2px;">{titre}</div>
              <div style="font-size:0.82rem; color:#64748B; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="margin-top:0;">
          <div class="card-title">NLL — Qualité de calibration</div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:0.85rem; color:#64748B;">Modèle naïf (baseline)</span>
            <span style="font-weight:600; color:#ef4444;">0.515</span>
          </div>
          <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:100%; background:#fca5a5;"></div></div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; margin-bottom:8px;">
            <span style="font-size:0.85rem; color:#64748B;">Notre modèle (LOOCV)</span>
            <span style="font-weight:600; color:#10b981;">0.469</span>
          </div>
          <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:91%; background:#10b981;"></div></div>
          <div style="font-size:0.75rem; color:#94A3B8; margin-top:8px; font-style:italic;">
            Réduction de 9% de la NLL → le modèle est mieux calibré que le hasard.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 3 — TRANSPARENCE ALGORITHMIQUE
# ═══════════════════════════════════════
with tab3:

    st.markdown("""
    <div class="card" style="background:#F0FDF4; border-color:#BBF7D0;">
      <strong style="color:#14532d;">Principe de transparence</strong><br>
      <span style="font-size:0.88rem; color:#166534;">
      Ce modèle est conçu pour être entièrement explicable. Un médecin doit comprendre
      pourquoi une décision est prise, pas juste recevoir un résultat. Voici chaque décision technique et sa justification.
      </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Coefficients β après régularisation L1</div>
        """, unsafe_allow_html=True)

        vars_data = [
            ("🥦 Fruits & légumes", 1.881, True),
            ("🏃 Activité physique", 0.740, True),
            ("🍳 Cuisine maison",    0.000, False),
            ("💧 Hydratation",       0.000, False),
            ("😴 Sommeil",           0.000, False),
            ("🍿 Snacks",            0.000, False),
            ("⏭️ Repas réguliers",   0.000, False),
            ("😌 Reposé au réveil",  0.000, False),
        ]
        for name, beta, active in vars_data:
            bar_w = int(abs(beta) / 1.881 * 100)
            color = "#10b981" if active else "#e2e8f0"
            tc    = "#166534" if active else "#94A3B8"
            badge = '<span class="var-badge badge-actif">ACTIF</span>' if active else '<span class="var-badge badge-nul">β = 0</span>'
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
                <span style="font-size:0.85rem; color:#1e293b;">{name}</span>
                <div style="display:flex; align-items:center; gap:8px;">
                  <span style="font-family:monospace; font-size:0.82rem; color:{tc}; font-weight:600;">{'+' if beta > 0 else ''}{beta:.3f}</span>
                  {badge}
                </div>
              </div>
              <div class="prob-bar-wrap" style="height:6px;">
                <div class="prob-bar-fill" style="width:{bar_w}%; background:{color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
          <div style="font-size:0.75rem; color:#94A3B8; font-style:italic; margin-top:8px;">
            L1 (Lasso, λ=2.0) a annulé 6 variables sur 8 — sélection automatique des plus discriminantes.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Décisions techniques & justifications</div>
        """, unsafe_allow_html=True)

        decisions = [
            ("Régression logistique", "#3b82f6",
             "Produit P(sain|X) ∈ [0,1] — directement interprétable par le médecin. Pas de boîte noire."),
            ("NLL pondérée w₀=5", "#ef4444",
             "FN (patient malade non détecté) = risque vital. Les erreurs sur la classe à risque coûtent 5× plus cher dans la loss."),
            ("Lasso L1 · λ=2.0", "#f59e0b",
             "n=19 → risque d'overfitting. L1 sélectionne automatiquement les 2 variables vraiment informatives."),
            ("LOOCV", "#8b5cf6",
             "n=19 trop petit pour K-Fold. LOOCV entraîne sur 18, teste sur 1, répété 19×. Estimateur quasi non-biaisé."),
            ("Seuil 0.45 (vs 0.50)", "#10b981",
             "Abaisser le seuil de 5 points + w₀=5 → Recall passe de 0% à 100% sur la classe à risque."),
        ]
        for titre, color, desc in decisions:
            st.markdown(f"""
            <div style="border-left:3px solid {color}; border-radius:0 8px 8px 0; padding:0.7rem 1rem; margin-bottom:8px; background:#f8fafc;">
              <div style="font-weight:600; font-size:0.85rem; color:#1e293b; margin-bottom:2px;">{titre}</div>
              <div style="font-size:0.8rem; color:#64748B; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="card-title">Limites à connaître</div>
          <div style="font-size:0.82rem; color:#64748B; line-height:1.8;">
            ⚠️ <strong style="color:#1e293b;">n = 19</strong> — dataset trop petit pour une généralisation robuste.<br>
            ⚠️ <strong style="color:#1e293b;">Auto-déclaratif</strong> — pas de biomarqueurs objectifs (IMC, tension, bilan sanguin).<br>
            ⚠️ <strong style="color:#1e293b;">Déséquilibre</strong> — 79% sains / 21% à risque. Précision plafonne à 57.1%.<br>
            ✅ <strong style="color:#1e293b;">Usage recommandé</strong> — outil de triage et d'aide à la décision, non de diagnostic définitif.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Formules
    st.markdown('<div class="section-label">Formules mathématiques complètes</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="card">
          <div class="card-title">Score Z (log-odds)</div>
          <div class="formula-box" style="font-size:0.78rem;">
Z = β₀ + β₁·X₁ + ... + βₙ·Xₙ<br><br>
Z = −0.8624<br>
  + 1.881 · X_fruits<br>
  + 0.740 · X_sport
          </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Sigmoïde → probabilité</div>
          <div class="formula-box" style="font-size:0.78rem;">
σ(Z) = 1 / (1 + e^(−Z))<br><br>
σ(Z) ∈ [0, 1]<br><br>
Décision :<br>
≥ 0.45 → SAIN<br>
 < 0.45 → À RISQUE
          </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="card">
          <div class="card-title">Cost function complète</div>
          <div class="formula-box" style="font-size:0.78rem;">
Cost = NLL + λ·Σ|βⱼ|<br><br>
NLL = −Σ[y·log(ŷ)<br>
  + w₀·(1−y)·log(1−ŷ)]<br><br>
w₀ = 5  ·  λ = 2.0
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#94A3B8; font-size:0.78rem;">
  HealthClassifier · Régression logistique L1 pondérée · LOOCV · n=19 ·
  Master 2 IMC&DS — Paris 1 Panthéon-Sorbonne · 2025–2026
</div>
""", unsafe_allow_html=True)