import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import streamlit as st
import pandas as pd

def display_roc_curve():
    st.header("Curva ROC")
    st.write("La Curva ROC (Receiver Operating Characteristic) es una representación gráfica de la capacidad de un clasificador binario.")

    data = pd.read_csv('Archivos csv/test_df.csv')
    data2 = pd.read_csv('Archivos csv/predictions.csv')
    # Ejemplo de datos de prueba y predicciones
    y_test = data['level']
    y_score = data2['0']

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)

if __name__ == "__main__":
    display_roc_curve()
