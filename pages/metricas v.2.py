import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la aplicación
st.title('Métricas de entrenamiento y validacion')

# Cargar datos
data = pd.read_csv('Archivos csv/training.csv')

# Definir las métricas que deseas graficar
metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1_score']

# Agrupar gráficos relacionados
with st.expander("Pérdida"):
    graf = pd.DataFrame({
        'Epoch': data['epoch'],
        'Entrenamiento': data['loss'],
        'Validación': data['val_loss']
    })
    st.line_chart(graf, use_container_width=True)

# Agrupar gráficos relacionados
with st.expander("Métricas de Clasificación"):
    metrics.append('lr')  # Añadir lr a la lista de métricas
    num_metrics = len(metrics)
    num_rows = num_metrics // 2 + num_metrics % 2  # Calcular el número de filas necesarias
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 10))  # Aumentar el tamaño del gráfico

    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2

        # Verificar si es lr para graficar solo una línea
        if metric == 'lr':
            axes[row, col].plot(data['epoch'], data[metric], label='Learning Rate', color='green', linestyle='-')
        else:
            axes[row, col].plot(data['epoch'], data[metric], label='Entrenamiento', color='blue', linestyle='-')
            axes[row, col].plot(data['epoch'], data[f'val_{metric}'], label='Validación', color='orange', linestyle='-')

        axes[row, col].set_title(metric.capitalize(), fontsize=14)  # Aumentar el tamaño del título
        axes[row, col].set_xlabel('Epoch', fontsize=12)  # Aumentar el tamaño de la etiqueta del eje x
        axes[row, col].set_ylabel(metric.capitalize(), fontsize=12)  # Aumentar el tamaño de la etiqueta del eje y
        axes[row, col].legend(fontsize=10)  # Aumentar el tamaño de la leyenda
        axes[row, col].grid(True, linewidth=1.5)  # Aumentar el tamaño de la cuadrícula

    # Ajustar el espacio entre los subgráficos
    plt.tight_layout()

    # Usar el ancho completo del contenedor para mostrar el gráfico
    st.pyplot(fig, use_container_width=True)










