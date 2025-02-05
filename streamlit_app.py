

import subprocess
import sys
import pkg_resources

def install_requirements():
    try:
        with open("requirements.txt", "r") as file:
            required_packages = [line.strip() for line in file.readlines() if line.strip()]

        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        missing_packages = [pkg for pkg in required_packages if pkg.split("==")[0] not in installed_packages]

        if missing_packages:
            print(f"📦 Installation des packages manquants : {missing_packages}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("✅ Toutes les dépendances sont maintenant installées.")
        else:
            print("✅ Toutes les dépendances sont déjà installées.")

    except Exception as e:
        print(f"⚠️ Erreur lors de la vérification/installation des dépendances : {e}")

# Exécuter la fonction
install_requirements()

import subprocess
import sys
import pkg_resources  
import streamlit as st
import pandas as pd
import math
from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import unidecode

# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse de l'Espérance de Vie - OMS",
    page_icon="🌍",
    layout="wide"
)

# --- import du dataset ---
path = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")  # ou spécifie le chemin complet si le fichier est ailleurs
dataset_file = f"{path}/Life Expectancy Data.csv"
df = pd.read_csv(dataset_file)

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
    Ce projet à pour objectif de définir les facteurs influençant l'espérance de vie à travers différents pays et années.
                """)
left_co, cent_co, last_co = st.columns([1, 3, 1])
with cent_co:
    st.markdown("""
                 ## **Hypothése:**
                 Peut on définir l'éspérence de vie d'une personne en fonction d'un ensemble d'éléments donnés?
                """)
    
st.markdown("""
            
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



# Aperçu du dataset
st.subheader("Aperçu du dataset BRUT")
st.write(df.head())


# Présentation des colonnes du dataset
st.subheader("Présentation des colonnes du dataset")

left_co, center_co, right_co = st.columns([1,2,1])
with center_co:
    st.markdown("""
                ### ⚠️ note: 
                **Life expectancy** sera notre variable **" Y "**. 
                il s'agit de l'élément que l'on souhaite prédire.
                """)

left_co, center_co, right_co = st.columns(3)
with left_co:
    st.markdown("""
        1. **Life expectancy** : L'espérance de vie moyenne (en années) pour le pays et l'année donnés.
        2. **Country** : Le pays correspondant aux données.
        3. **Year** : L'année où les données ont été collectées.
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

Voici ce que nous pouvons examiner :
""")
chart_type = st.radio("Choisissez le type de graphique :", 
                      ["1. Taille du dataset", "2. Types de données par colonne", "3. Présence de valeurs manquantes", "4. Statistiques descriptives des variables numériques"])

# Taille du dataset
if chart_type == "1. Taille du dataset":
    st.markdown("## **Taille du dataset**")
    st.write(f"**Nombre de lignes** : {df.shape[0]}")
    st.write(f"**Nombre de colonnes** : {df.shape[1]}")

# Types de données par colonne
if chart_type == "2. Types de données par colonne":
    left_co, right_co = st.columns(2)
    with left_co:
        st.markdown("""
            ## **Types de données par colonne**
            """)
        st.write(df.dtypes)
    with right_co:
        st.markdown("""
                    ## **Observations**
                    - **titre des colone** nous devons formatter les titres des collones.
                    - les variables de type "object" (catégoriques) doivent être converties en valeurs numériques pour que les modèles puissent les utiliser correctement.
                    - **nettoyage et normalisation des données** Nous devons formatez les lignes dans les colonnes pour êtres sur qu'il n'y ai pas d'erreur de saisie (Espace, Majuscule...) 
                    - ⚠️ **country**: Initialement, nous souhaitions utiliser la méthode **Encodage One-Hot Encoding**.
                    comme nous allons le voir juste aprés, cette méthode n'est pas viable dans notre cas.
                    - **status** : Ici, nous utiliserons la méthode **Label Encoding**.
                    """)
        # --- Génération du Heatmap des Corrélations ---
if chart_type == "2. Types de données par colonne":
    left_co, center_co, right_co = st.columns([1,3,1])
    with center_co:   
        st.markdown("""
                    ### Sélection des Variables pour Identifier le Pays:
                    ⚠️ **Country** : Initialement, je souhaitais utiliser la méthode **One-Hot Encoding**, qui crée une colonne pour chaque pays et assigne une valeur de 0 ou 1 en fonction de l'appartenance de la ligne à un pays.  
                    
                    ⚠️ **Cependant** : cette méthode génère un trop grand nombre de colonnes, ce qui entraîne une **malédiction de la dimensionnalité**.  

                    Je vais donc chercher **deux variables faiblement corrélées entre elles**, mais significatives pour différencier les pays (par exemple, le PIB s'il est disponible). Ensuite, nous créerons une **nouvelle variable** en effectuant une opération entre ces deux variables :  
                    `nouvelle_variable = variable1 / (variable2 + 1)`.  

                    Pour cela, une **heatmap** nous aidera à identifier les deux variables les plus pertinentes.            
                    """)


        # Sélectionner les colonnes numériques
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Vérifier qu'on a bien des colonnes numériques
        if not numeric_columns:
            st.warning("⚠️ Aucune colonne numérique disponible pour la corrélation.")
        else:
            # Calcul de la matrice de corrélation
            correlation_matrix = df[numeric_columns].corr()

            # Création du heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5, center=0)
            plt.title("Matrice de Corrélation des Variables")
            st.pyplot(fig)

            # Sélectionner deux variables avec une **faible corrélation** entre elles
            st.markdown("### 📉 Sélection de Variables Faiblement Corrélées")

            # Conversion de la matrice de corrélation en DataFrame pour l'analyse
            corr_pairs = correlation_matrix.unstack().reset_index()
            corr_pairs.columns = ["Variable 1", "Variable 2", "Corrélation"]

            # Éviter les doublons et les corrélations parfaites (1.0 avec soi-même)
            corr_pairs = corr_pairs[corr_pairs["Variable 1"] != corr_pairs["Variable 2"]]
            corr_pairs["Corrélation"] = corr_pairs["Corrélation"].abs()  # Valeur absolue pour éviter les signes
            corr_pairs = corr_pairs.sort_values("Corrélation")  # Trier par corrélation croissante

            # Affichage des **10 paires de variables les moins corrélées**
            st.write("Top 10 des paires de variables avec une faible corrélation (meilleur choix pour encoder le pays) :")
            st.write(corr_pairs.head(10))

            # --- Génération de la nouvelle variable country_index ---
            # Nettoyage des noms de colonnes : suppression des espaces avant et après
            # Nettoyage des noms de colonnes : suppression des espaces
            df.columns = df.columns.str.strip()

            # Vérification que les colonnes nécessaires sont bien présentes
            if "Total expenditure" in df.columns and "HIV/AIDS" in df.columns:
                # Création de la colonne country_index
                df["country_index"] = df["Total expenditure"] / (df["HIV/AIDS"] + 1)  # Évite la division par zéro

                # Mettre à jour les données nettoyées dans st.session_state
                st.session_state["df_cleaned"] = df

                st.success("✅ Nouvelle colonne `country_index` créée avec succès !")
                st.write("📌 **Colonnes disponibles après la création** :", df.columns.tolist())

                # Affichage pour vérifier si la colonne est bien ajoutée
                st.write(df[["Country", "Total expenditure", "HIV/AIDS", "country_index"]].head())

                df = df.drop(columns=["Country"])

        

# Présence de valeurs manquantes
if chart_type == "3. Présence de valeurs manquantes":
    st.markdown("## **Présence de valeurs manquantes**")
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_data = pd.DataFrame({
        "Valeurs manquantes": missing_values,
        "Pourcentage (%)": missing_percent
    })
    st.write(missing_data)

if chart_type == "4. Statistiques descriptives des variables numériques":
    st.markdown("## **Statistiques descriptives des variables numériques**")
    
    # Calcul des statistiques descriptives
    descriptive_stats = df.describe().transpose()

    # Ajout d'un format plus lisible
    descriptive_stats = descriptive_stats.rename(columns={
        "count": "Nombre de valeurs",
        "mean": "Moyenne",
        "std": "Écart-type",
        "min": "Valeur min",
        "25%": "1er quartile (Q1)",
        "50%": "Médiane (Q2)",
        "75%": "3e quartile (Q3)",
        "max": "Valeur max"
    })

    # Affichage sous forme de tableau interactif
    st.write(descriptive_stats)

# ----------------------------------------------------
#  présentation cleaning
# ----------------------------------------------------
def clean_dataset(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    for col in ["country", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].apply(lambda x: unidecode.unidecode(x))  # Supprimer accents
    return df

def imput_null(df):
    columns_median = ["life_expectancy", "adult_mortality", "alcohol", "total_expenditure", 
                "thinness_5-9_years", "thinness__1-19_years", "bmi", ]
    for col in columns_median:
        if col in df.columns:  # Vérifier si la colonne existe avant d'appliquer
            df[col].fillna(df[col].median(), inplace=True)
    # Moyenne pour les indicateurs de vaccination (plus de groupby, uniquement moyenne globale)
    columns_vaccination = ["hepatitis_b", "polio", "diphtheria"]
    for col in columns_vaccination:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)  
    # Moyenne pour les indicateurs économiques et d'éducation
    columns_mean = ["income_composition_of_resources", "schooling"]
    for col in columns_mean:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    # Imputation basée sur la médiane globale pour GDP et Population (pas de groupby)
    columns_gdp_population = ["gdp", "population", "bmi", "thinness__1-19_years", "thinness_5-9_years" ]
    for col in columns_gdp_population:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

df = clean_dataset(df)

# radio:
left_co, center_co, right_co = st.columns([1,3, 1])
with center_co:
    st.markdown(""" # DATA-CLEANSING """)
    chart_type = st.radio(" ", 
                      ["Typages des Données", "Label Encoding (status)", "Remplacement des valeurs manquantes"])
    if chart_type == "Typages des Données":
        st.markdown(""" 
                ### **Typages des Données** 
                Ici, nous formatons le titre des colones, ainsi que toutes les lignes afin d'étre sur d'avoir une nomenclature identique pour chaque colonne, et pour chaque objet.
                """)
        st.write(df.head(10).round(2))
        df = df.round(2)
    
    if chart_type == "Label Encoding (status)":
        st.markdown(""" 
                ### **Label Encoding (status)** 
                Ici, nous transformons les valeurs string des status en 0 ou 1 en fonction de la valeurs initiale du status.
                """)
        from sklearn.preprocessing import LabelEncoder

        # Initialiser le label encoder
        label_encoder = LabelEncoder()

        # Transformer la colonne Status en 0 et 1
        df["status_encoded"] = label_encoder.fit_transform(df["status"])

        # Vérifier le mapping des valeurs
        status_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("Mapping des valeurs :", status_mapping)

        # Supprimer l'ancienne colonne si nécessaire
        df = df.drop(columns=["status"])
        st.write(df.head())


if chart_type == "Remplacement des valeurs manquantes":
    col_right, col_left = st.columns(2)
    with col_right:
        st.markdown(""" 
                ### **Remplacement des valeurs manquantes** 
                Les valeures manquantes peuvent impacter la qualité de l'analyse et fausser les modèles prédictifs.  
                Voici une analyse détaillée des colonnes concernées et les stratégies retenues pour leur traitement.
            ---

            ## 🛠 Méthodes d'imputation retenues :
            - **Médiane** : utilisée pour éviter l'effet des valeurs extrêmes (outliers).
            - **Moyenne par pays** : préférée pour les indicateurs influencés par le pays (vaccination, éducation).
            - **Médiane par pays & année** : pour les variables comme la **Population** et le **PIB**, très hétérogènes.

            ---
                    
            ## Colonnes avec valeurs manquantes et stratégies de traitement
                    
            ---

            | **Colonne**                      | **% de valeurs manquantes** | **Statistiques clés** | **Stratégie d'imputation** |
            |----------------------------------|---------------------------|-----------------------|----------------------------|
            | **Life expectancy**              | 0.34%                     | Médiane = 72.1, Écart-type = 9.52 | **Médiane** : évite l'impact des outliers. |
            | **Adult Mortality**              | 0.34%                     | Médiane = 144, Écart-type = 124.29 | **Médiane** : car forte dispersion des valeurs. |
            | **Alcohol**                      | 6.60%                     | Médiane = 3.75, Écart-type = 4.05 | **Médiane** : variable à forte variabilité. |
            | **Hepatitis B**                  | 18.82%                    | Médiane non fournie | **Moyenne par pays** : taux de vaccination. |
            | **BMI**                          | 1.15%                     | Médiane = 22.4 | **Médiane** : homogénéité relative des valeurs. |
            | **Polio**                        | 0.65%                     | Médiane non fournie | **Moyenne par pays** : taux de vaccination. |
            | **Total expenditure**             | 7.69%                     | Médiane = 5.85 | **Médiane** : évite les extrêmes. |
            | **Diphtheria**                   | 0.65%                     | Médiane non fournie | **Moyenne par pays** : taux de vaccination. |
            | **GDP**                          | 15.25%                    | Médiane = 3457, Écart-type = 15,312 | **Médiane par pays** : très forte dispersion du PIB. |
            | **Population**                   | 22.19%                    | Médiane non fournie | **Médiane par pays & année** : donnée très variable. |
            | **Thinness 1-19 years**          | 1.15%                     | Médiane = 2.3 | **Médiane** : variable peu dispersée. |
            | **Thinness 5-9 years**           | 1.15%                     | Médiane = 2.3 | **Médiane** : même raison que Thinness 1-19 ans. |
            | **Income composition of resources** | 5.68%                  | Médiane = 0.62 | **Moyenne par pays** : reflète les disparités économiques. |
            | **Schooling**                    | 5.55%                     | Médiane = 12.3 | **Moyenne par pays** : car impact fort du pays. |

            ---
                            
            """)
        with col_left:
            st.write("")
            st.markdown("""
                        lignes aprés traitements des valeurs manquantes
                        """)
            imput_null(df)
            st.write(df.isnull().sum())

imput_null(df)


    # ---------------------------------------------
# --- Interface utilisateur ---
st.subheader("🔍 Visualisation interactive des données")

# Liste des colonnes numériques pour l’analyse
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Sélection du type de graphique
col_left, col_cent, col_right = st.columns([2,2,2])
st.markdown("""
    ### Analyser en détail les différentes variables du dataset afin de mieux comprendre leur distribution et d'identifier d'éventuelles anomalies.
    **Sélection interactive d'une colonne**
    Choix d'une colonne numérique parmi celles du dataset pour afficher ses statistiques et visualiser sa distribution.

    **Affichage d'un histogramme et d'un boxplot**
    L'histogramme permet d'observer la fréquence des valeurs et leur répartition, tandis que le boxplot met en évidence la dispersion des données et détecte les valeurs extrêmes.

    **Détection automatique des outliers**
    Les valeurs aberrantes sont identifiées à l'aide de la méthode de l'IQR (Interquartile Range), qui repère les valeurs situées en dehors de la plage normale des données.

    **Option pour exclure les outliers**
    Nous avons si besoin la possibilité d'exclure les valeurs aberrantes afin d'observer leur impact sur la distribution et d'analyser uniquement les données considérées comme représentatives.
    """)   

# --- Exploration avancée des données ---
st.markdown("---")

if "df_cleaned" not in st.session_state:
    st.session_state["df_cleaned"] = df.copy()  # On garde une copie propre des données

# Sélection d'une colonne pour l'analyse
numeric_columns = st.session_state["df_cleaned"].select_dtypes(include=['float64', 'int64']).columns.tolist()
selected_column = st.selectbox("Sélectionnez une colonne :", numeric_columns)
col_1, col_2, col_3 = st.columns([2,2,2])
with col_2:
    st.write(f"## {selected_column}")

col_1, col_2, col_3 = st.columns([1,2,2])
with col_1:
    if selected_column:
        # Afficher les statistiques descriptives
        st.write(f"### 📊 Statistiques descriptives")
        st.write(st.session_state["df_cleaned"][selected_column].describe())
with col_2:
    if selected_column:
        # Histogramme avec KDE
        st.write(f"### 📈 Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(st.session_state["df_cleaned"][selected_column], bins=50, kde=True, ax=ax)
        plt.xlabel(selected_column)
        plt.ylabel("Fréquence")
        st.pyplot(fig)

with col_3:
    if selected_column:
        # Boxplot pour identifier les outliers
        st.write(f"### 📊 Boxplot")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=st.session_state["df_cleaned"][selected_column], ax=ax)
        plt.xlabel(selected_column)
        st.pyplot(fig)
col_1, col_2, col_3 = st.columns([2,2,2])
with col_2:
    st.write(f"### Détection des outliers avec IQR: {selected_column}")

col_1, col_2 = st.columns([2,4])
with col_1:
    # Détection des outliers avec IQR
    Q1 = st.session_state["df_cleaned"][selected_column].quantile(0.25)
    Q3 = st.session_state["df_cleaned"][selected_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = st.session_state["df_cleaned"][
        (st.session_state["df_cleaned"][selected_column] < lower_bound) |
        (st.session_state["df_cleaned"][selected_column] > upper_bound)
    ]

    # Afficher les outliers détectés
    st.write(f"### 🔎 Nombre de valeurs aberrantes détectées dans {selected_column} : {outliers.shape[0]}")
    if not outliers.empty:
        st.write(outliers[[selected_column]])

    # Option pour exclure les outliers
   # Option pour exclure les outliers avec une clé unique
exclude_outliers = st.checkbox("Exclure les valeurs aberrantes", key=f"exclude_{selected_column}")

with col_2:
    if exclude_outliers:
        # Supprimer les outliers du dataset stocké dans `st.session_state`
        st.session_state["df_cleaned"] = st.session_state["df_cleaned"][
            (st.session_state["df_cleaned"][selected_column] >= lower_bound) & 
            (st.session_state["df_cleaned"][selected_column] <= upper_bound)
        ]
        
        st.write(f"### 📉 Nouvelle distribution après suppression des outliers ({st.session_state['df_cleaned'].shape[0]} valeurs restantes)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(st.session_state["df_cleaned"][selected_column], bins=50, kde=True, ax=ax)
        plt.xlabel(selected_column)
        plt.ylabel("Fréquence")
        st.pyplot(fig)

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        #  corrélation des variable
col_1,col_2 = st.columns([2,1])
# --- Analyse des corrélations ---
st.markdown("---")
with col_1:
    st.markdown("##  Analyse des Corrélations entre les Variables")

    # --- HEATMAP INTERACTIVE AVEC SEUIL DE CORRÉLATION ---
    st.markdown("### 🔥 Heatmap Interactive des Corrélations avec Seuil Ajustable")

    # Sélection des colonnes numériques
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Calcul de la matrice de corrélation complète
    correlation_matrix = df[numeric_columns].corr()

    # Curseur interactif pour filtrer les corrélations
    correlation_threshold = st.slider("🔍 Sélectionnez le seuil de corrélation à afficher :", 
                                    min_value=0.0, 
                                    max_value=1.0, 
                                    value=0.0,  # Par défaut, afficher tout
                                    step=0.05)

    # Filtrage des corrélations en fonction du seuil sélectionné
    filtered_corr_matrix = correlation_matrix.copy()
    mask = abs(filtered_corr_matrix) < correlation_threshold
    filtered_corr_matrix[mask] = 0  # Mettre à zéro les valeurs en dessous du seuil

    # Création de la heatmap avec Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", center=0)
    plt.title(f"Carte de Chaleur des Corrélations (Seuil ≥ {correlation_threshold})")

    # Affichage dans Streamlit
    st.pyplot(fig)