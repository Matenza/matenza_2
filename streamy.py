import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Lire le fichier CSV
energi = pd.read_csv('Expresso_churn_dataset.csv', nrows=155000)

# Titre principal de l'application
st.title("Analyse de Churn")

# Titre de la barre latérale
st.sidebar.title('Menu de Navigation')

# Définir les différentes pages de l'application
pages = ['Présentation', 'Nettoyage', 'Visualisation et Entraînement', 'Interprétation']
page = st.sidebar.radio('Aller à la page', pages)

# Présentation
if page == pages[0]:
    st.write("# Informations sur la DataFrame")
    st.write("Le  concours de prédiction du taux de désabonnement d'Expresso est organisé par l'une des plus grandes sociétés de télécommunications du Sénégal, Expresso, afin de déterminer la probabilité qu'un client « se désabonne », c'est-à-dire qu'un client régulier cesse d'utiliser les services de l'entreprise (sans effectuer aucune transaction) après 90 jours. Par conséquent, l'objectif du concours est de développer un modèle prédictif qui tenterait de trouver de manière préventive les clients qui seront les plus susceptibles de devenir inactifs en fonction des données de formation fournies par l'entreprise. Par rapport à nos autres concours d'IA, celui-ci est un peu différent dans le fait qu'il s'agit d'une tâche de prédiction binaire, qui s'appuiera fortement sur le côté données.")
    st.write("Il faut noter qu'il y a pas mal de données - en fait, 25 millions d'entrées - à disposition des équipes, mais l'efficacité de l'algorithme réside dans le prétraitement et le nettoyage de la partie données. Par rapport aux autres compétitions, celle-ci se déroule sans aucun algorithme de base ni aucune aide fournie par les hôtes, et le seul moyen possible de communiquer avec les autres est via le forum de discussion. Cependant, ce projet est relativement facile à calculer avec un équipement moderne, donc la difficulté repose davantage sur le côté algorithme que sur le matériel. Enfin, pour cette compétition, notre équipe vise à utiliser l'apprentissage profond plutôt que l'apprentissage automatique pour obtenir de meilleurs résultats.")
    st.image("Home_Page.png")
# Nettoyage des données
elif page == pages[1]:
    st.write("## Nettoyage des données")
    
    # Afficher la DataFrame initiale
    st.write("### DataFrame initiale")
    st.dataframe(energi)
    
    # Encodage des variables catégorielles
    st.write("### Encodage des variables catégorielles")
    for col in energi.select_dtypes(include=['object']).columns:
        if col != 'user_id':  # Exclure 'user_id' de l'encodage
            m = LabelEncoder()
            energi[col] = m.fit_transform(energi[col])
            energi[col] = energi[col].astype('category')
    
    # Afficher la DataFrame après encodage
    st.write("### DataFrame après encodage")
    st.dataframe(energi)
    
    # Supprimer des colonnes inutiles
    st.write("### Suppression des colonnes inutiles")
    energi.drop(['user_id', 'REGION', 'TENURE', 'MRG', 'TOP_PACK'], axis=1, inplace=True)
    
    # Remplir les valeurs manquantes avec la moyenne de la colonne
    st.write("### Remplissage des valeurs manquantes")
    energi.fillna(energi.mean(), inplace=True)
    
    # Afficher la DataFrame après nettoyage
    st.write("### DataFrame après nettoyage")
    st.dataframe(energi)
    
    # Informations sur les données
    if st.checkbox('Afficher des informations sur les données'):
        st.write("### Informations sur les données")
        st.write(energi.info())
        st.write(energi.describe())
    
    # Détection de valeurs manquantes
    if st.checkbox('Détection de valeurs manquantes'):
        st.write("### Détection de valeurs manquantes")
        st.dataframe(energi.isnull().sum())

# Visualisation et Entraînement
elif page == pages[2]:
    
    
   
    for col in energi.select_dtypes(include=['object']).columns:
        if col != 'user_id':  # Exclure 'user_id' de l'encodage
            m = LabelEncoder()
            energi[col] = m.fit_transform(energi[col])
            energi[col] = energi[col].astype('category')
    

    
  
    energi.drop(['user_id', 'REGION', 'TENURE', 'MRG', 'TOP_PACK'], axis=1, inplace=True)
    
    
    energi.fillna(energi.mean(), inplace=True)
    

    # Entraînement du modèle KMeans
    st.write("### Entraînement avec KMeans")
    model = KMeans(n_clusters=5)
    model.fit(energi)
    labels = model.predict(energi)
    
    # Visualisation des clusters
    st.write("### Visualisation des clusters")
    plt.figure(figsize=(10, 6))
    plt.scatter(energi.iloc[:, 0], energi.iloc[:, 1], c=labels, s=50, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    st.pyplot(plt)
    
    # Visualisation de l'inertie
    st.write("### Visualisation de l'inertie")
    inertie = []
    ranger = range(1, 20)
    for k in ranger:
        kmeans = KMeans(n_clusters=k).fit(energi)
        inertie.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranger, inertie)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    st.pyplot(plt)

# Interprétation des résultats
elif page == pages[3]:
    st.write("## Interprétation des résultats")
    # Ajoutez du code pour l'interprétation des résultats ici
    st.write("## À la lumière de cette analyse nous constatons des habitudes assez similaire  chez la plupart des utilisateurs ")
