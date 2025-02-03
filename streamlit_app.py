

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
                    - ** titre des colone** nous devons formatter les titres des collones.
                    - les variables de type "object" (catégoriques) doivent être converties en valeurs numériques pour que les modèles puissent les utiliser correctement.
                    - **nettoyage et normalisation des données** Nous devons formatez les lignes dans les colonnes pour êtres sur qu'il n'y ai pas d'erreur de saisie (Espace, Majuscule...) 
                    - **country**: nous allons utiliser la méthode **Encodage One-Hot Encoding** qui va creer une collone pour chaque pays et assigner une valeur 0 ou 1 en fonction de l'appartenance de la ligne à un pays. """)
    

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
df_onehot = pd.get_dummies(df, columns=["country","status"]).round(2)

# radio:
left_co, center_co, right_co = st.columns([1,3, 1])
with center_co:
    st.markdown(""" # DATA-CLEANSING """)
    chart_type = st.radio(" ", 
                      ["Typages des Données", "One-Hot Encoding (country, status)", "Remplacement des valeurs manquantes"])
    if chart_type == "Typages des Données":
        st.markdown(""" 
                ### **Typages des Données** 
                Ici, nous formatons le titre des colones, ainsi que toutes les lignes afin d'étre sur d'avoir une nomenclature identique pour chaque colonne, et pour chaque objet.
                """)
        st.write(df.head(10).round(2))
        df = df_onehot.round(2)
    
    if chart_type == "One-Hot Encoding (country, status)":
        st.markdown(""" 
                ### **One-Hot Encoding (country, status)** 
                Ici, nous transformons les valeurs string des pays et des status en colonne par pays, avec 0 ou 1 en fonction de l'appartenace de la ligne au pays.
                """)
        st.write(df_onehot.head(10))

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
            imput_null(df_onehot)
            st.write(df_onehot.isnull().sum())

imput_null(df_onehot)
df = df_onehot

    # ---------------------------------------------
# --- Interface utilisateur ---
st.subheader("🔍 Visualisation interactive des données")

# Liste des colonnes numériques pour l’analyse
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Sélection du type de graphique
chart_type = st.radio("Choisissez le type de graphique :", 
                      ["Histogramme", "Heatmap", "Boxplots"])

if chart_type == "Histogramme":
    col_left, col_cent, col_right = st.columns([1,3,2])
    with col_cent:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        st.subheader("Histogramme interactif")
        if "year" in numeric_columns:
            numeric_columns.remove("year")
        selected_column = st.selectbox("Sélectionnez une colonne :", numeric_columns)

        if selected_column:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.histplot(df[selected_column], bins=30, kde=True, ax=ax)
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
    
    with col_right:st.markdown("""
            ### Observation des valeurs aberrantes et des valeurs extrêmes :
            **income_composition_of_resources** : Dans l'histogramme, nous pouvons observer une valeur qui semble aberrante.  
            Lorsque l'on sélectionne cette colonne dans le boxplot, nous constatons effectivement qu'une donnée à 0 semble étrange.  
            Cette valeur apparaît également dans l'analyse descriptive et dans le boxplot.  

            Nous pouvons en déduire qu'il manque probablement un ensemble de données, car il est peu probable d'avoir un `income_composition_of_resources`
            égal à zéro, même dans des cas extrêmes.  
            Je choisis ici d'exclure les outliers.
            """)
        
        # note: gerer les outliners à partir de ici, reprendre chat gpt
 if chart_type == "Boxplots":
    import matplotlib.pyplot as plt
    import streamlit as st
    import pandas as pd

    # 🔹 Liste des colonnes à nettoyer (éviter les doublons dans la liste)
    columns_to_clean = list(set([
        'schooling', 'income composition of resources', 'adult mortality', 'hiv/aids', 'bmi', 'diphtheria',
        'polio', 'infant deaths', 'alcohol', 'percentage expenditure', 'hepatitis b', 'measles', 'under-five deaths',
        'gdp', 'population', 'thinness  1-19 years', 'thinness 5-9 years'
        ]))

    # 🔹 Fonction pour nettoyer les outliers
    def clean_outliers(data, columns):
        for col in columns:
            if col in data.columns:
                # Calcul des quartiles et de l'IQR
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                # Définir les bornes inférieure et supérieure
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Remplacer les valeurs extrêmes par les bornes
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound

        return data

    # 🔹 Appliquer le nettoyage des outliers AVANT affichage des graphiques
    df_cleaned = clean_outliers(df.copy(), columns_to_clean)

    # 🔹 Récupérer la liste des colonnes disponibles après nettoyage
    available_columns = df_cleaned.columns.tolist()

    # 🔹 Sélection des variables à afficher
    st.subheader(" Distribution des variables avec Boxplots (Nettoyées)")
    selected_columns = st.multiselect("Sélectionnez les variables :", columns_to_clean, default=columns_to_clean[:5])

    if selected_columns:
        # Vérifier la correspondance des noms de colonnes après nettoyage
        selected_columns_cleaned = [col.lower().replace(' ', '_') for col in selected_columns if col.lower().replace(' ', '_') in available_columns]

        # Nombre de colonnes et lignes pour affichage
        n_cols = 3
        n_rows = len(selected_columns_cleaned) // n_cols + (len(selected_columns_cleaned) % n_cols > 0)

        # Création des subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))

        # Gestion du cas où une seule colonne est sélectionnée
        if n_rows == 1:
            axes = [axes]

        # Générer les boxplots pour chaque colonne sélectionnée
        for i, col in enumerate(selected_columns_cleaned):
            row, col_index = divmod(i, n_cols)
            ax = axes[row, col_index] if n_rows > 1 else axes[col_index]

            ax.boxplot(df_cleaned[col].dropna(), vert=False, patch_artist=True, showmeans=True)
            ax.set_title(selected_columns[i])  # Garder le titre original
            ax.set_xlabel('Valeurs')

        # Supprimer les axes inutilisés
        for j in range(len(selected_columns_cleaned), n_rows * n_cols):
            row, col_index = divmod(j, n_cols)
            fig.delaxes(axes[row, col_index] if n_rows > 1 else axes[col_index])

        # Ajuster l'affichage
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Veuillez sélectionner au moins une variable.")


if chart_type == "Heatmap":
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader(" Heatmap des Corrélations")

    # Sélection des colonnes numériques
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Vérifier que des colonnes numériques existent
    if not numeric_columns:
        st.warning("Aucune colonne numérique disponible pour la heatmap.")
    else:
        # Sélecteur pour le seuil de corrélation
        correlation_threshold = st.slider("Seuil de corrélation (Valeurs absolues supérieures)", 0.0, 1.0, 0.2, 0.05)

        # Calcul de la matrice de corrélation
        correlation_matrix = df[numeric_columns].corr()

        # Filtrer les corrélations selon le seuil
        filtered_corr = correlation_matrix[abs(correlation_matrix) > correlation_threshold]

        # Création de la heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)

        # Affichage dans Streamlit
        st.pyplot(fig)


# --------------------------------------------------------------


# Section Transformation des données
st.markdown("---")
st.markdown("## 5. Transformation des données")

st.write("""
Tranformons les données pour les rendre exploitables dans mes modèles.  
Cela inclut :
- **Normaliser** ou **standardiser** les données si nécessaire.
- **Encoder** les variables catégoriques.
- **Créer ou combiner** des colonnes pour extraire des informations pertinentes.
""")

st.markdown("---")
st.markdown("###  Standardisation des colonnes en vue d'une régression linéaire")
st.write("""
La **standardisation** est préférable à la normalisation dans le cadre de données distribuées,  
notamment pour garantir une meilleure interprétation et stabilité des modèles de **régression linéaire**.
""")
# --------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

#  Liste des colonnes numériques à standardiser
columns_to_standardize = [
    'adult mortality', 'infant deaths', 'alcohol', 'percentage expenditure',
    'hepatitis b', 'measles', 'bmi', 'under-five deaths',
    'polio', 'gdp', 'population', 'thinness  1-19 years',
    'thinness 5-9 years', 'income composition of resources', 'schooling'
]


# Interface utilisateur pour activer ou non la standardisation
st.markdown("###  Standardisation des colonnes")
st.write("La standardisation est appliquée aux variables numériques afin de les rendre comparables.")

apply_standardization = st.checkbox("Appliquer la standardisation")

if apply_standardization:
    # Vérifier si les colonnes existent dans le dataset après nettoyage
    columns_to_standardize = [col for col in columns_to_standardize if col in df.columns]

    if columns_to_standardize:
        # Initialiser le standard scaler
        scaler = StandardScaler()

        # Appliquer la standardisation
        df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

        st.success(" Standardisation appliquée avec succès !")

        # Affichage des statistiques après transformation
        st.write("###  Statistiques après standardisation")
        st.write(df[columns_to_standardize].describe())
#---------------------------------------------------------------
st.markdown("---")
st.markdown("## 6. Analyse des données")

st.write("""
Analisons les données pour identifier des relations significatives entre les variables.  
Cela inclut: 
Calculer des **corrélations** entre les variables ainsi que
Réaliser des **tests statistiques** pour valider mes hypothèses.
""")

import streamlit as st
import pandas as pd

# Vérification et affichage des corrélations avec 'life expectancy'
st.markdown("###  Corrélations avec 'Life expectancy'")

if 'life expectancy' in df.columns:
    # Calcul des corrélations
    correlations = df.corr()['life expectancy'].sort_values(ascending=False)

    # Affichage dans un tableau interactif
    st.write(" **Top des corrélations avec l'espérance de vie** :")
    st.write(correlations)
else:
    st.warning("⚠ La colonne 'life expectancy' n'existe pas dans le dataset.")
