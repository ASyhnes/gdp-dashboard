# Prédiction de l'Espérance de Vie - Analyse de Données de l'OMS

Ce projet est une application interactive développée avec **Streamlit** permettant d'explorer et de prédire l'espérance de vie en fonction de divers facteurs socio-économiques et de santé. Il s'appuie sur un **dataset de l'OMS** et met en œuvre des techniques de **machine learning** pour analyser les tendances à travers différents pays et années.

---

## Objectifs du projet

🔹 **Nettoyage et prétraitement des données** : Suppression des valeurs manquantes et transformation des variables.  
🔹 **Exploration et visualisation** : Identifier les tendances et relations entre les variables.  
🔹 **Modélisation prédictive** : Implémentation de modèles de machine learning pour estimer l'espérance de vie.  
🔹 **Interface interactive** : Développement d'une application avec **Streamlit** pour permettre une exploration dynamique des données.  

---

## 📂 Structure du projet

📦 espérance_vie_ml 
│── 📁 data # Contient le dataset de l'OMS 
│── 📁 notebooks # Analyses exploratoires et modèles ML 
│── 📁 src # Code source de l'application 
│── ─ streamlit_app.py # Application Streamlit principale 
│── ─ requirements.txt # Dépendances Python
│── ─ README.md # Documentation du projet


---

## Installation et utilisation

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/ASyhnes/gdp-dashboard.git
cd gdp-dashboard
```

### 2️⃣ Installer les dépendances
Python 3.8+, puis installez les bibliothèques nécessaires :

´´´
pip install -r requirements.txt
´´´

### 3️⃣ Lancer l'application
Exécutez la commande suivante pour démarrer Streamlit :

´´´
streamlit run streamlit_app.py
´´´

L'interface web sera accessible à l'adresse http://localhost:8501.

## Fonctionnalités de l'application
✅ Exploration des données : Affichage des tendances d'espérance de vie par pays et par année.
✅ Visualisation interactive : Graphiques dynamiques illustrant l'évolution des facteurs clés.
✅ Prédiction avec ML : Utilisation d'un modèle pour estimer l'espérance de vie en fonction des entrées utilisateur.

## Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.



