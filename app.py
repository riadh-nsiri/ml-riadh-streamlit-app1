import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns

# Machine learning pakages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import datetime

def main():
    html_temp = """
            <div style="background-color:{};padding:10px;border-radius:10px">
            <h1 style="color:{};text-align:center;"> Application de Machine Learning  </h1>
            </div>
            """
    st.markdown(html_temp.format('royalblue', 'white'), unsafe_allow_html=True)
    today = st.date_input("La date: ", datetime.datetime.now())

    activities = ["Statistiques Exploratoires", "Graphiques", "Modèles", "A propos"]
    choice = st.sidebar.selectbox("Selectionner une activité : ", activities)

    if choice == 'Statistiques Exploratoires':
        st.subheader("Analyse exploratoire des données")

        data = st.file_uploader("Télécharger vos données sous format csv ou txt : ", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Nombre des lignes et des colonnes"):
                st.write(df.shape)

            if st.checkbox("Les colonnes en Json"):
                all_columns = df.columns.tolist()
                st.write(all_columns)

            if st.checkbox("Selectionner une ou plusieurs variables"):
                selected_columns = st.multiselect("Selectionner des variables", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Les statistiques Descriptives"):
                st.write(df.describe())

            if st.checkbox("Les valeurs de compatge"):
                st.write(df.iloc[:, -1].value_counts())

            if st.checkbox("Matrice de corrélation"):
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot()

            if st.checkbox("Diagramme circulaire"):
                all_columns = df.columns.tolist()
                columns_to_plot = st.selectbox("Selectionner une colonne", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()

    elif choice == 'Graphiques':
        st.subheader("Visualisation des données")

        data = st.file_uploader("Télécharger vos données sous format csv ou txt : ", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Selectionner le type de graphique", ["Aires", "Barres", "Courbe", "Histogramme"])
            selected_columns_names = st.multiselect("Selectionner colonne de graphique", all_columns_names)

        if st.button("Genérer votre graphique"):
            st.success("Genérer votre graphique personalisé de {} pour la variable {}".format(type_of_plot, selected_columns_names))

            if type_of_plot == 'Aires':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'Barres':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

            elif type_of_plot == 'Courbe':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            elif type_of_plot == 'Histogramme':
                cust_data = df[selected_columns_names]
                fig, ax = plt.subplots()
                ax.hist(cust_data, bins=2)
                st.pyplot(fig)

            elif type_of_plot:
                cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()

    elif choice == 'Modèles':
        st.subheader("L'apprentissage automatique")

        data = st.file_uploader("Télécharger vos données sous format csv ou txt : ", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7

            models=[]
            models.append(("Regression Logistique", LogisticRegression()))
            models.append(("Analyse discriminante linéaire", LinearDiscriminantAnalysis()))
            models.append(("Algorithme des k-plus proches voisins", KNeighborsClassifier()))
            models.append(('Arbres de Décision', DecisionTreeClassifier()))
            models.append(('Classifieur Bayésien Naif', GaussianNB()))
            models.append(('Machine à vecteurs de support', SVC()))

            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'

            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())

                accuracy_results = {"modèle": name, "modèle d'apprentissage": cv_results.mean(), "écart-type": cv_results.std()}
                all_models.append(accuracy_results)

            if st.checkbox("Modélisation en tableaux : "):
                st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std), columns=["Nom du modèle","Modèle d'Apprentissage Automatique", "Écart type"]))

            if st.checkbox("Modélisation en Json"):
                st.json(all_models)

    elif choice == 'A propos':
        st.subheader("A propos")
        st.text("Vous pouvez utiliser cet outil: avec un fichier csv avec séparateur virgule, Riadh NSIRI ")

    html_temp = """
            <div style="background-color:#808080 ;padding:5px;border-radius:10px;font-weight: bolder">
            <h6 style="color:{};text-align:center;"> © 2021 . R. NSIRI </h6>
            </div>
            """
    st.markdown(html_temp.format('royalblue', 'white'), unsafe_allow_html=True)


if __name__ == '__main__':
    main()