import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    st.set_page_config(
        page_title="Grad-CAM Visualization",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Grad-CAM (Class Activation Maps)")
    st.write('Visualiza qué áreas de una imagen están siendo consideradas por el modelo para hacer una predicción:')
    st.header("Descripción del Grad-CAM")
    st.write("Grad-CAM (Gradient-weighted Class Activation Mapping) es una técnica de visualización que resalta las áreas importantes de una imagen para que un modelo de red neuronal haga una predicción.")

    col1, col2, _ = st.columns([3, 5, 2])

    with col1:
        # Seleccionar modelo y capa convolucional
        model_path = st.selectbox('Seleccionar modelo:', [
            'models/BestModel-densenet121_preprocessed-retinal.h5',
            'models/epoch1-densenet121_preprocessed-retinal.h5',
            'models/epoch2-densenet121_preprocessed-retinal.h5',
            'models/epoch3-densenet121_preprocessed-retinal.h5',
            'models/epoch4-densenet121_preprocessed-retinal.h5',
            'models/epoch5-densenet121_preprocessed-retinal.h5',
            'models/epoch7-densenet121_preprocessed-retinal.h5',
            'models/epoch8-densenet121_preprocessed-retinal.h5'
        ])
        model = load_model(model_path)

        # Cargar nombres de capas convolucionales desde un archivo CSV
        conv_layers = load_conv_layers('Archivos csv/model_layers_output_shapes.csv')
        if conv_layers:
            last_conv_layer_name = st.selectbox('Nombre de la última capa convolucional:', conv_layers)
        else:
            st.error('No se pudieron cargar las capas convolucionales.')

        # Cargar imagen
        image_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

    with col2:
        if image_file is not None:
            image_path = image_file.name
            image, img_array = load_image(image_file)

            if image is not None and img_array is not None:
                st.image(image, caption="Imagen cargada", use_column_width=True)

                if st.button("Generar Grad-CAM"):
                    # Generar Grad-CAM
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                    if heatmap is not None:
                        # Convertir heatmap a un tipo de datos compatible (CV_8UC1) para aplicar colormap
                        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                        # Redimensionar heatmap_colored para que coincida con las dimensiones de la imagen original
                        heatmap_colored_resized = cv2.resize(heatmap_normalized, (image.shape[1], image.shape[0]))

                        # Aplicar colormap y ajustar la intensidad de la superposición
                        alpha = st.slider('Intensidad Grad-CAM', 0.0, 1.0, 0.6)
                        beta = 1.0 - alpha  # Resto para que la suma de alpha y beta sea 1

                        heatmap_colored = cv2.applyColorMap(heatmap_colored_resized, cv2.COLORMAP_JET)

                        # Ajustar el rango de intensidades del mapa de calor para mejorar la visibilidad
                        heatmap_colored = cv2.normalize(heatmap_colored, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                        # Superponer Grad-CAM en la imagen original con la intensidad ajustada
                        superimposed_img = cv2.addWeighted(image, alpha, heatmap_colored, beta, 0)

                        # Mostrar Grad-CAM
                        st.image(superimposed_img, caption='Grad-CAM', use_column_width=True)
                    else:
                        st.error('No se pudo generar el Grad-CAM.')
            else:
                st.error('Error: No se pudo cargar la imagen.')

@st.cache_data
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_image(image_file, target_size=(224, 224)):
    image = Image.open(image_file)
    image = image.convert('RGB')  # Asegurarse de que la imagen esté en formato RGB
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, target_size)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.array(img_array, dtype=np.float32)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return np.array(image), img_array

@st.cache_data
def load_conv_layers(csv_path):
    try:
        conv_layers_df = pd.read_csv(csv_path)
        return conv_layers_df['Layer Name'].tolist()
    except Exception as e:
        st.error(f'Error al cargar las capas convolucionales: {e}')
        return []

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    except ValueError as e:
        st.error(f'Error al obtener la capa {last_conv_layer_name}: {e}')
        return None

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        st.error(f'Error: no se pudieron calcular los gradientes para la capa {last_conv_layer_name}')
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

if __name__ == "__main__":
    main()
