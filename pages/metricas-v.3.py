import streamlit as st
import pandas as pd

# Título de la aplicación
st.title('Métricas de entrenamiento y validación')

# Cargar datos
data = pd.read_csv('Archivos csv/training.csv')

# Definir las métricas que deseas graficar
metrics = ['Loss', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1_score']
val_metrics = ['Val_loss', 'Val_AUC', 'Val_accuracy', 'Val_precision', 'Val_recall', 'Val_F1_score']

# Lista desplegable para seleccionar la métrica
selected_metric = st.selectbox('Seleccionar métrica:', metrics)

# Crear el DataFrame para el gráfico
graf = pd.DataFrame({
    'Epoch': data['epoch'],
    'Entrenamiento': data[selected_metric.lower()],
    'Validación': data[val_metrics[metrics.index(selected_metric)].lower()]
})

# Mostrar el gráfico
st.line_chart(graf, x='Epoch', y=['Entrenamiento', 'Validación'])
