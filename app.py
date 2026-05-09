import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, log_loss
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
st.set_page_config(
    page_title="HealthClassifier — Aide au diagnostic",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #f8fafc; }

/* Header médical */
.med-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 32px 40px;
    border-radius: 16px;
    margin-bottom: 28px;
    color: white;
}
.med-header h1 { font-size: 28px; font-weight: 700; margin: 0 0 6px 0; }
.med-header p  { font-size: 14px; opacity: 0.8; margin: 0; }
.med-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 4px 12px; border-radius: 20px;
    font-size: 12px; margin-bottom: 14px;
}

/* Cards */
.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.card-title {
    font-size: 13px; font-weight: 600;
    color: #64748b; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 16px;
}

/* Résultat verdict */
.verdict-sain {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #16a34a;
    border-radius: 16px; padding: 28px;
    text-align: center; margin: 20px 0;
}
.verdict-risque {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #dc2626;
    border-radius: 16px; padding: 28px;
    text-align: center; margin: 20px 0;
}
.verdict-vigilance {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid #d97706;
    border-radius: 16px; padding: 28px;
    text-align: center; margin: 20px 0;
}
.verdict-title { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
.verdict-prob  { font-size: 52px; font-weight: 800; line-height: 1; margin: 12px 0; }
.verdict-sub   { font-size: 14px; opacity: 0.75; }

/* Recommandation */
.reco-box {
    border-radius: 10px; padding: 16px 20px;
    margin-top: 16px; font-size: 14px; line-height: 1.6;
}
.reco-sain    { background: #f0fdf4; border-left: 4px solid #16a34a; color: #166534; }
.reco-risque  { background: #fff1f2; border-left: 4px solid #dc2626; color: #991b1b; }
.reco-vigilance { background: #fffbeb; border-left: 4px solid #d97706; color: #92400e; }

/* KPI metric */
.kpi-box {
    background: white; border-radius: 12px;
    padding: 20px; text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-num  { font-size: 32px; font-weight: 800; color: #1e3a5f; line-height: 1; }
.kpi-lbl  { font-size: 12px; color: #64748b; margin-top: 6px; font-weight: 500; }
.kpi-sub  { font-size: 11px; color: #94a3b8; margin-top: 3px; }

/* Variable toggle */
.var-card {
    background: white; border-radius: 10px;
    padding: 16px 18px; margin-bottom: 10px;
    border: 1px solid #e2e8f0;
    transition: all 0.2s;
}
.var-name { font-size: 14px; font-weight: 600; color: #1e293b; }
.var-impact { font-size: 11px; color: #64748b; margin-top: 2px; }

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: #f1f5f9;
    padding: 6px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 20px;
    font-weight: 500; font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Info box */
.info-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    font-size: 13px; color: #1e40af; margin: 12px 0;
}

/* Footer */
.footer {
    text-align: center; color: #94a3b8;
    font-size: 12px; padding: 24px 0;
    border-top: 1px solid #e2e8f0; margin-top: 40px;
}

/* Barre de probabilité */
.prob-bar-bg {
    background: #e2e8f0; border-radius: 99px;
    height: 12px; margin: 8px 0; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.5s ease;
}

/* Contribution factors */
.factor-row {
    display: flex; align-items: center;
    gap: 12px; padding: 10px 0;
    border-bottom: 1px solid #f1f5f9;
}
.factor-name { font-size: 13px; color: #374151; flex: 1; }
.factor-bar-bg {
    width: 120px; height: 8px;
    background: #e2e8f0; border-radius: 99px; overflow: hidden;
}
.factor-bar-pos { background: #16a34a; height: 100%; border-radius: 99px; }
.factor-bar-neu { background: #94a3b8; height: 100%; border-radius: 99px; }
.factor-val { font-size: 12px; font-weight: 600; width: 50px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── DATA & MODEL ──
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
    X, y = df[FEATS].values, df['y'].values
    model = LogisticRegression(penalty='l1', C=0.5, class_weight={0:5,1:1},
                                solver='liblinear', max_iter=2000, random_state=42)
    model.fit(X, y)
    loo = LeaveOneOut()
    cv_probs = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:,1]
    cv_preds = (cv_probs >= 0.45).astype(int)
    return model, X, y, cv_probs, cv_preds

model, X, y, cv_probs, cv_preds = load_and_train()

def sigmoid(z): return 1 / (1 + np.exp(-z))

# ── HEADER ──
st.markdown("""
<div class="med-header">
    <div class="med-badge">🏥 Outil d'aide à la décision médicale — Paris 1 Panthéon-Sorbonne</div>
    <h1>HealthClassifier</h1>
    <p>Régression logistique L1 pondérée · Entraîné sur 19 profils · Validé par Leave-One-Out Cross-Validation<br>
    Objectif : estimer la probabilité qu'un patient soit en bonne santé à partir de ses habitudes de vie déclarées.</p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3 = st.tabs([
    "🩺  Évaluation patient",
    "📊  Performance du modèle",
    "📋  À propos de l'étude"
])

# ════════════════════════════════════════════
# TAB 1 — ÉVALUATION PATIENT
# ════════════════════════════════════════════
with tab1:

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="card-title">📝 Habitudes de vie du patient</div>', unsafe_allow_html=True)
        st.markdown("Renseignez les réponses du patient aux questions suivantes :")

        st.markdown("---")

        # ── Variables ACTIVES (L1 les a gardées)
        st.markdown("**🎯 Facteurs déterminants** *(poids statistique élevé)*")

        fruits = st.radio(
            "🥦 Consommation de fruits & légumes par jour",
            ["✅  Plus de 2 portions / jour", "❌  Moins de 2 portions / jour"],
            help="Corrélation r=0.68 avec la bonne santé — facteur n°1 dans notre dataset"
        )

        sport = st.radio(
            "🏃 Activité physique par semaine (> 30 min)",
            ["✅  3 séances ou plus", "❌  Moins de 3 séances"],
            help="Corrélation r=0.61 — facteur n°2 dans notre dataset"
        )

        st.markdown("---")
        st.markdown("**📋 Autres habitudes** *(collectées, non retenues par le modèle L1)*")

        c1, c2 = st.columns(2)
        with c1:
            st.radio("⏭️ Saute des repas ?",   ["Non", "Oui"],
                     disabled=True, help="β=0 après régularisation L1")
            st.radio("💧 Eau / jour ?",         ["+ de 1,5L", "- de 1,5L"],
                     disabled=True, help="β=0 après régularisation L1")
            st.radio("🍳 Cuisine maison ?",     ["Oui", "Non"],
                     disabled=True, help="β=0 après régularisation L1")
        with c2:
            st.radio("😴 Heures de sommeil ?",  ["+ de 7h", "- de 7h"],
                     disabled=True, help="β=0 après régularisation L1")
            st.radio("😌 Se sent reposé ?",     ["Oui", "Non"],
                     disabled=True, help="β=0 après régularisation L1")
            st.radio("🍿 Snacks entre repas ?", ["Rarement", "Souvent"],
                     disabled=True, help="β=0 après régularisation L1")

        st.caption("ℹ️ Les variables grisées ont été automatiquement écartées par la régularisation Lasso (L1) — leur coefficient β a été mis à 0 car leur contribution n'est pas statistiquement discriminante sur ce dataset.")

    with col_result:
        # ── Calcul
        f_val = 1 if "Plus" in fruits else 0
        s_val = 1 if "3 séances" in sport else 0
        z     = -0.8624 + 1.881 * f_val + 0.740 * s_val
        prob  = sigmoid(z)
        pct   = round(prob * 100, 1)

        # ── Verdict
        if prob < 0.45:
            classe, couleur, emoji = "À risque", "risque", "🔴"
            reco = "⚠️ <b>Bilan approfondi recommandé.</b> Le profil du patient présente des habitudes défavorables à sa santé. Envisager un suivi renforcé et des conseils en hygiène de vie."
        elif prob < 0.65:
            classe, couleur, emoji = "Vigilance", "vigilance", "🟡"
            reco = "👁️ <b>Suivi préventif conseillé.</b> Le profil est intermédiaire. Encourager l'amélioration des habitudes alimentaires et sportives lors de la prochaine consultation."
        else:
            classe, couleur, emoji = "Bonne santé", "sain", "🟢"
            reco = "✅ <b>Suivi standard.</b> Le profil du patient est favorable. Maintenir les habitudes actuelles et prévoir un suivi de routine."

        st.markdown(f"""
        <div class="verdict-{couleur}">
            <div class="verdict-title">{emoji} {classe}</div>
            <div class="verdict-prob">{pct}%</div>
            <div class="verdict-sub">Probabilité estimée d'être en bonne santé<br>
            <small>Seuil de décision : 45% · Modèle L1 pondéré (w₀=5)</small></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="reco-box reco-{couleur}">{reco}</div>', unsafe_allow_html=True)

        # ── Barre visuelle
        st.markdown("**Probabilité P(sain | habitudes)**")
        bar_color = "#16a34a" if prob >= 0.65 else "#d97706" if prob >= 0.45 else "#dc2626"
        st.markdown(f"""
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%; background:{bar_color}"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:#94a3b8; margin-top:4px;">
            <span>0% — À risque</span>
            <span style="color:{bar_color}; font-weight:700;">{pct}%</span>
            <span>100% — Sain</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Contribution des facteurs
        st.markdown("---")
        st.markdown("**Contribution des facteurs au score**")

        factors = [
            ("🥦 Fruits & légumes", 1.881, f_val, "Fort · β=+1.881"),
            ("🏃 Activité physique", 0.740, s_val, "Modéré · β=+0.740"),
        ]
        for name, beta, val, desc in factors:
            contrib = beta * val
            w = int(abs(contrib) / 1.881 * 100)
            color = "#16a34a" if val == 1 else "#dc2626"
            status = "✅ Favorable" if val == 1 else "❌ Défavorable"
            st.markdown(f"""
            <div class="factor-row">
                <span class="factor-name">{name}<br><small style="color:#94a3b8">{desc}</small></span>
                <div class="factor-bar-bg">
                    <div style="width:{w}%; background:{color}; height:100%; border-radius:99px;"></div>
                </div>
                <span class="factor-val" style="color:{color}">{status}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Calcul détaillé
        with st.expander("🔢 Voir le détail du calcul"):
            st.code(f"""
Score Z  = β₀ + β_fruits × X_fruits + β_sport × X_sport
         = -0.8624 + 1.881 × {f_val} + 0.740 × {s_val}
         = {z:.4f}

P(sain)  = σ(Z) = 1 / (1 + e^(-{z:.4f}))
         = {prob:.4f}  →  {pct}%

Décision : {pct}% {'≥' if prob >= 0.45 else '<'} 45% → {classe.upper()}
            """, language="text")

        st.markdown(f"""
        <div class="info-box">
        ℹ️ <b>Interprétation médicale :</b> Ce score est un indicateur statistique basé sur 19 profils auto-déclarés. Il ne remplace pas le jugement clinique du médecin. Un score élevé ne garantit pas la bonne santé ; un score faible ne confirme pas une pathologie.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — PERFORMANCE DU MODÈLE
# ════════════════════════════════════════════
with tab2:

    st.markdown("### Performance du modèle — Leave-One-Out Cross-Validation (n=19)")
    st.markdown("Le modèle a été évalué en LOOCV : entraîné sur 18 patients, testé sur 1, répété 19 fois.")

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("100%",  "Recall · À risque",    "0 patient manqué"),
        ("57.1%", "Précision · À risque", "4/7 alertes justifiées"),
        ("80.8%", "F1-Score Macro",        "Équilibre global"),
        ("0.80",  "AUC-ROC",              "vs 0.50 aléatoire"),
        ("0.469", "NLL · LOOCV",          "vs 0.515 baseline"),
    ]
    for col, (n, l, s) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-num">{n}</div>
                <div class="kpi-lbl">{l}</div>
                <div class="kpi-sub">{s}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matrice de confusion")
        st.markdown("""
| | **Prédit : À risque** | **Prédit : Sain** |
|:---|:---:|:---:|
| **Réel : À risque** (n=4) | ✅ **4** Vrais Négatifs | ❌ 0 Faux Positifs |
| **Réel : Sain** (n=15) | ⚠️ 3 Faux Négatifs | ✅ **12** Vrais Positifs |
        """)
        st.success("**Recall = 100%** : les 4 patients à risque sont tous détectés. Zéro cas manqué.")
        st.warning("**3 Faux Négatifs** : 3 patients sains classés 'à risque'. Examens supplémentaires inutiles mais sans danger.")
        st.markdown("""
        <div class="info-box">
        ⚕️ <b>Choix médical assumé :</b> En contexte médical, rater un patient malade (FN) est beaucoup plus grave qu'une fausse alarme (FP). On a donc pondéré les erreurs : w₀=5 (à risque) vs w₁=1 (sain). Résultat : Recall = 100% sur la classe critique.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Pourquoi ces métriques ?")
        metrics = [
            ("Recall · À risque", "100%", "#16a34a",
             "Priorité absolue. FN = patient malade non détecté → risque vital. On ne peut pas se le permettre."),
            ("Précision · À risque", "57.1%", "#d97706",
             "4 alertes sur 7 sont justifiées. Acceptable — mieux vaut 3 examens inutiles que rater un malade."),
            ("F1-Score Macro", "80.8%", "#2d6a9f",
             "Équilibre Précision/Recall sur les deux classes. Robuste face au déséquilibre 79/21."),
            ("AUC-ROC", "0.80", "#7c3aed",
             "80% du temps, le modèle classe mieux un patient sain qu'un patient à risque, indépendamment du seuil."),
            ("Accuracy", "84.2%", "#64748b",
             "Non retenue comme critère principal — trompeuse : un modèle naïf (toujours sain) atteindrait 79%."),
        ]
        for name, val, color, desc in metrics:
            st.markdown(f"""
            <div style="display:flex; gap:14px; align-items:flex-start; padding:12px 0; border-bottom:1px solid #f1f5f9;">
                <div style="font-size:20px; font-weight:800; color:{color}; width:70px; flex-shrink:0;">{val}</div>
                <div>
                    <div style="font-size:13px; font-weight:600; color:#1e293b;">{name}</div>
                    <div style="font-size:12px; color:#64748b; margin-top:3px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Pourquoi LOOCV et pas K-Fold ?")
    c1, c2 = st.columns(2)
    with c1:
        st.error("**K-Fold K=5 — Écarté**\n\n→ ~4 observations de test par fold\n\n→ Potentiellement 0 cas 'à risque' dans un fold\n\n→ Variance de l'estimateur trop élevée avec n=19")
    with c2:
        st.success("**LOOCV — Retenu ✓**\n\n→ Entraîne sur 18 patients, teste sur 1\n\n→ Répété 19 fois — estimateur quasi non-biaisé\n\n→ Méthode optimale pour les petits datasets")


# ════════════════════════════════════════════
# TAB 3 — À PROPOS
# ════════════════════════════════════════════
with tab3:

    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown("### Contexte de l'étude")
        st.markdown("""
        Ce modèle a été conçu dans le cadre d'un projet de Machine Learning supervisé (Master 2 IMC&DS, Paris 1 Panthéon-Sorbonne, 2025–2026).

        **Problématique :** Un médecin traitant souhaite disposer d'un indicateur rapide pour identifier les patients dont les habitudes de vie suggèrent un profil à risque — sans attendre un bilan biologique complet.

        **Données :** 19 réponses collectées via Google Forms. Chaque répondant a déclaré ses habitudes (alimentation, sport, sommeil, hydratation) et son sentiment subjectif de bonne santé.

        **Variable cible :** Y=1 si le patient se déclare en bonne santé, Y=0 sinon.
        """)

        st.markdown("### Décisions techniques et justifications")
        decisions = [
            ("Régression logistique", "Problème binaire → P(Y=1|X) ∈ [0,1]. Interprétable par un non-expert. Coefficients directement lisibles."),
            ("Régularisation L1 (Lasso, λ=2)", "n=19 avec 8 variables → risque d'overfitting. L1 annule les variables non discriminantes (6 sur 8 → β=0). Modèle sparse et interprétable."),
            ("Pondération w₀=5", "FN (patient malade non détecté) = risque vital >> FP (fausse alarme). Les erreurs sur classe à risque coûtent 5× plus dans la NLL."),
            ("Seuil 0.45 (vs 0.50)", "Abaissé pour maximiser le Recall sur la classe à risque. Combiné à w₀=5 : Recall passe de 0% à 100%."),
            ("LOOCV", "n=19 trop petit pour K-Fold (variance trop élevée). LOOCV entraîne sur 18, teste sur 1, répété 19 fois — estimateur quasi non-biaisé."),
        ]
        for title, body in decisions:
            with st.expander(f"**{title}**"):
                st.markdown(body)

    with col2:
        st.markdown("### Variables du modèle")
        coef_df = pd.DataFrame({
            "Variable": ["🥦 Fruits & légumes","🏃 Sport","⏭️ Repas réguliers",
                         "🍳 Cuisine maison","💧 Hydratation","😴 Sommeil","😌 Reposé","🍿 Snacks"],
            "β": [1.881, 0.740, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            "Statut": ["✅ Actif","✅ Actif","❌ Annulé","❌ Annulé",
                       "❌ Annulé","❌ Annulé","❌ Annulé","❌ Annulé"],
            "Corr. EDA": [0.68, 0.61, -0.03, 0.28, 0.29, 0.23, 0.18, 0.23]
        })
        st.dataframe(coef_df, hide_index=True, use_container_width=True)

        st.markdown("### Limites à connaître")
        st.warning("""
**n=19** — Dataset très petit. Les IC 95% bootstrap incluent 0 pour la plupart des coefficients. Le recall 100% doit être validé sur un jeu indépendant.

**Auto-déclaratif** — La cible "je me sens en bonne santé" est subjective. Aucun biomarqueur objectif (IMC, tension, glycémie).

**Déséquilibre** — 79% sains / 21% à risque. La précision à risque (57.1%) reflète cette contrainte.
        """)

        st.markdown("### Pour aller plus loin")
        st.info("""
- Collecter n ≥ 100 réponses
- Intégrer des biomarqueurs objectifs
- Tester ElasticNet (L1+L2 combinés)
- Valider sur une population indépendante
        """)

# ── FOOTER ──
st.markdown("""
<div class="footer">
    HealthClassifier · Régression logistique L1 pondérée · LOOCV · n=19<br>
    Master 2 IMC&DS — Paris 1 Panthéon-Sorbonne · 2025–2026
</div>
""", unsafe_allow_html=True)