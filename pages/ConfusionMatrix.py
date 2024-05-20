import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def display_confusion_matrix():
    st.header("Matriz de Confusión")
    st.write("La matriz de confusión permite visualizar el desempeño del modelo clasificando cada clase.")

    data = pd.read_csv('Archivos csv/test_df.csv')
    data2 = pd.read_csv('Archivos csv/predictions.csv')

    data['level'] = data['level'].astype(int)
    data2['0'] = data2['0'].astype(int)



    # Ejemplo de datos de prueba y predicciones
    y_test = data['level']
    y_pred = data2['0']

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    st.pyplot(plt)

if __name__ == "__main__":
    display_confusion_matrix()
