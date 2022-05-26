import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import * 
from sklearn.metrics import *

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(page_title='The Machine Learning Web Application',
    layout='wide')
def main():
    st.title("The Machine Learning Web App")
    st.sidebar.markdown('The ML Web Application')

    def load_data():
        data = pd.read_csv('data.csv')
        bmi_mean = data['bmi'].mean()
        data['bmi'] = data['bmi'].fillna(bmi_mean)
        data.drop(columns=['id'], inplace=True)
        data['age'] = data['age'].astype('int32')
        categorical_variables = data.select_dtypes(include=['object']).columns.to_list()
        data.drop(data[data.gender == 'Other'].index, axis = 0, inplace=True)
        label_encoder = preprocessing.LabelEncoder()
        for feature in categorical_variables:
            data[feature]= label_encoder.fit_transform(data[feature])
            data[feature].unique()
        return data

    data = load_data()
    dataset = st.sidebar.selectbox("Choose to see dataset",('Original dataset', 'After cleaning & preprocessing'))



    # data2 = data.copy()
    if dataset == 'Original dataset':
        if st.sidebar.button('Click to see', key='click2'):
            st.subheader("")
            st.dataframe(pd.read_csv('data.csv'))
    if dataset == 'After cleaning & preprocessing':
        if st.sidebar.button('Click to see', key='click'):
            st.subheader("")
            st.dataframe(data)

    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, X_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()
        if 'CountPlot' in metrics_list:
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(x='stroke', data=data, hue='gender')
            st.pyplot(fig)


    algorithm = st.sidebar.selectbox("Select Algorithm", ("KMeans", "RandomForest", "Linear Regression"))

    if algorithm == "KMeans":
        X = data
        ncluster = st.sidebar.selectbox('Choose number of clusters?',(2, 3))
        if ncluster == 2:
            first = st.sidebar.selectbox("Choose one column from the table?",(data.columns))
            second = st.sidebar.selectbox("Choose one column from the table?",(data.columns), key = "<uniquevalue>")
        if ncluster == 3:
            first = st.sidebar.selectbox("Choose one column from the table?",(data.columns))
            second = st.sidebar.selectbox("Choose one column from the table?",(data.columns), key = "<uniquevalue>")
            third = st.sidebar.selectbox("Choose one column from the table?",(data.columns), key = "<unique>")
        cluster = st.sidebar.selectbox("What view to plot?",('Before clustering', 'After clustering'))
        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("")
            kmeans = KMeans(n_clusters=ncluster, random_state=0)
            kmeans.fit(X)
            y_kmeans = kmeans.predict(X)
            centers = kmeans.cluster_centers_
            if ncluster == 2:
                if cluster == 'Before clustering':
                    fig = plt.figure(figsize=(10, 4))
                    plt.scatter(X[first], X[second], s=50)
                    st.pyplot(fig)
                else:
                    fig = plt.figure(figsize=(10, 4))
                    plt.scatter(X[first], X[second], c=y_kmeans, s=50, cmap='viridis')
                    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
                    st.pyplot(fig)
            if ncluster == 3:
                if cluster == 'Before clustering':
                    fig = plt.figure(figsize=(10, 4))
                    plt.scatter(X[first], X[second], X[third])
                    st.pyplot(fig)
                else:
                    fig = plt.figure(figsize=(10, 4))
                    plt.scatter(X[first], X[second], X[third], c=y_kmeans, cmap='viridis')
                    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
                    st.pyplot(fig)
    if algorithm == 'Linear Regression':
        test_size = st.sidebar.slider('Train-test split %', 1, 100)
        first_variable = st.sidebar.selectbox("Choose one column from the table?",(data.columns))
        second_variable = st.sidebar.selectbox("Choose one column from the table?",(data.columns), key = "<uniquevalueofsomesort>")
        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("")
            X = np.array(data[first_variable]).reshape(-1, 1)
            y = np.array(data[second_variable]).reshape(-1, 1)  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
  
            regr = LinearRegression()
            regr.fit(X_train, y_train)

            y_pred = regr.predict(X_test)
            fig = plt.figure(figsize=(10,4))
            plt.scatter(X_test, y_test, color ='b')
            plt.plot(X_test, y_pred, color ='k')
            plt.xlabel(first_variable)
            plt.ylabel(second_variable)
            plt.legend()
            st.pyplot(fig)


    if algorithm  == 'RandomForest':
        test_size = st.sidebar.slider('Train-test split %', 1, 100)
        n_estimators  = st.sidebar.number_input("The number of trees", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        plot = st.sidebar.selectbox("Which one to plot?",('Confusion Matrix', 'Precision-Recall Curve','CountPlot'))
        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            X = data.drop(columns = ['stroke'], axis=1)
            y = data['stroke']
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=0)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            plot_metrics(plot)



if __name__ == '__main__':
    main()


