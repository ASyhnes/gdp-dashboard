import streamlit as st
import pandas as pd
import math
from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns


# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse de l'Espérance de Vie - OMS",
    page_icon="🌍",
    layout="wide"
)

# --- import du dataset ---
# Définir le chemin d'accès au dataset
file_path = "Life Expectancy Data.csv"  # ou spécifie le chemin complet si le fichier est ailleurs

# Chargement des données via kaglle
path = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")

# --- Texte de présentation ---
left_co, cent_co, last_co = st.columns([1, 3, 1])
imgleft_co, imgcent_co, imglast_co = st.columns(3)
with cent_co:
    st.title(" Analyse de l'Espérance de Vie - OMS")
with imgcent_co:
    st.image("images/oms_logo.png", width=200)

st.write("")
st.markdown("""---""")

# --- chapitre sur présenttation du dataset ---
left_co, cent_co, last_co = st.columns([1, 3, 1])
with cent_co:
    st.markdown("""
                 # **Prédiction de l'espérance de vie**
                """)
st.markdown("""
Ce projet utilise un **dataset de l'OMS** pour analyser les facteurs influençant l'espérance de vie à travers différents pays et années.

🔎 **Objectifs** :
- Explorer les facteurs clés qui influencent la longévité.
- Nettoyer et préparer les données.
- Comparer différents modèles de prédiction.
- Présenter les résultats de manière interactive
            
📌 **Méthodologie** :
1. **Exploration des données** 
2. **Nettoyage des valeurs manquantes** 
3. **Modélisation avec Machine Learning** 
4. **Interprétation des résultats** 
        

---
""", unsafe_allow_html=True)

left_co, cent_co, last_co = st.columns([1, 3, 1])
with cent_co:
    st.markdown("""
                 ## **hypothése:**
                 Peut on définir l'éspérence de vie d'une personne en fonction d'un ensemble d'éléments données?
                """)


# Chargement des données
file_path = path + "/Life Expectancy Data.csv"
df = pd.read_csv(file_path)

# Aperçu du dataset
st.subheader("Aperçu du dataset")
st.write(df.head())


# Présentation des colonnes du dataset
st.subheader("Présentation des colonnes du dataset")


left_co, center_co, right_co = st.columns(3)
with left_co:
    st.markdown("""
        1. **Country** : Le pays correspondant aux données.
        2. **Year** : L'année où les données ont été collectées.
        3. **Life expectancy** : L'espérance de vie moyenne (en années) pour le pays et l'année donnés.
        4. **Adult Mortality** : Taux de mortalité des adultes (probabilité de décès entre 15 et 60 ans, par 1000 habitants).
        5. **Infant deaths** : Nombre de décès d'enfants de moins de 1 an pour 1000 naissances vivantes.
        6. **Alcohol** : Consommation moyenne d'alcool (litres par habitant par an).
        7. **Percentage expenditure** : Dépenses publiques pour la santé (% du PIB).
        """, unsafe_allow_html=True)
    with center_co:
        st.markdown("""
            8. **Hepatitis B** : Taux de couverture vaccinale contre l'hépatite B chez les enfants (%).
            9. **Measles** : Nombre de cas signalés de rougeole.
            10. **BMI** : Indice de masse corporelle moyen (IMC) pour la population.
            11. **Under-five deaths** : Nombre de décès d'enfants de moins de 5 ans pour 1000 naissances vivantes.
            12. **Polio** : Taux de couverture vaccinale contre la poliomyélite chez les enfants (%).
            13. **Total expenditure** : Dépenses totales pour la santé (% du PIB).
            14. **Diphtheria** : Taux de couverture vaccinale contre la diphtérie chez les enfants (%).
            """, unsafe_allow_html=True)
    with right_co:
        st.markdown("""
            15. **HIV/AIDS** : Décès dus au VIH/sida pour 1000 habitants.
            16. **GDP** : Produit intérieur brut par habitant (USD).
            17. **Population** : Population totale du pays.
            18. **Thinness 1-19 years** : Prévalence de la maigreur chez les enfants et adolescents (1 à 19 ans) (%).
            19. **Thinness 5-9 years** : Prévalence de la maigreur chez les enfants (5 à 9 ans) (%).
            20. **Income composition of resources** : Indicateur composite de revenus (entre 0 et 1).
            21. **Schooling** : Durée moyenne de scolarisation (années).
            """, unsafe_allow_html=True)

#  --- statistique desctiptives: 
st.markdown("""
            ---
            """)

left_co, center_co, right_co = st.columns(3)
# Analyse descriptive des données
with center_co:
    st.subheader("Analyse descriptive du dataset")

# Aperçu général
st.markdown("""
### **🔍 Analyse descriptive**
L'objectif de cette section est d'explorer les statistiques de base du dataset et d'identifier les premières observations clés.

Voici ce que nous allons examiner :
1. Taille du dataset
2. Types de données par colonne
3. Présence de valeurs manquantes
4. Statistiques descriptives des variables numériques
""")

# Taille du dataset
st.write(f"**Nombre de lignes** : {df.shape[0]}")
st.write(f"**Nombre de colonnes** : {df.shape[1]}")

# Types de données par colonne
left_co, right_co = st.columns(2)
with left_co:
    st.markdown("### **Types de données**")
    st.write(df.dtypes)

# Présence de valeurs manquantes
with right_co:
    st.markdown("### **Valeurs manquantes**")
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_data = pd.DataFrame({
        "Valeurs manquantes": missing_values,
        "Pourcentage (%)": missing_percent
    })
    st.write(missing_data)

left_co, center_co, right_co = st.columns([1,3, 1])
with center_co:
    st.markdown(""" 
                ## **Observation et traitement sur les types de valeurs et les valeurs manquantes** """)
    st.markdown(""" 
                ### **Typages des Données** """)
