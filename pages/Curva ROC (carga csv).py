import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import streamlit as st
import pandas as pd

def display_roc_curve():
    st.header("Curva ROC")
    st.write("La Curva ROC (Receiver Operating Characteristic) es una representación gráfica de la capacidad de un clasificador binario.")

    uploaded_files = st.file_uploader("Cargar archivos CSV", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        test_file, pred_file = uploaded_files[0], uploaded_files[1]
        test_df = pd.read_csv(test_file)
        pred_df = pd.read_csv(pred_file)

        if 'level' in test_df.columns and '0' in pred_df.columns:
            y_test = test_df['level']
            y_score = pred_df['0']

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
        else:
            st.error("Los archivos CSV deben contener columnas 'level' y '0'.")


if __name__ == "__main__":
    display_roc_curve()
