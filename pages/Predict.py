import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from utils import load_ben_color, predict_imagen

def main():
    st.set_page_config(
        page_title="Detección de Retinopatía Diabética",
        page_icon="👁️",
        layout="wide",
    )

    st.title("Detección de Retinopatía Diabética")
    st.write("---")
    st.write("Sube una imagen para que el modelo haga una predicción.")
    st.header("Algún texto que quede bien aquí")
    st.write("Aquí se puede poner un texto")

    col1, col2, _ = st.columns([3, 5, 2])

    with col1:
        image_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            img = Image.open(image_file)
            st.image(img, caption="Imagen cargada", use_column_width=True)

    with col2:
        if image_file is not None:
            img_array = np.array(img)
            processed_image = load_ben_color(img_array)
            processed_image = np.expand_dims(processed_image, axis=0)
            st.write("Imagen preprocesada:")
            st.image(processed_image[0], caption="Imagen preprocesada", use_column_width=True)
            st.write("---")
            if st.button("Realizar Predicción de la categoría de la imagen"):
                pred = predict_imagen(processed_image)
                st.success('Éxito al realizar la predicción!')
                st.write('La categoría predicha para la imagen:')
                st.write(pred)

if __name__ == "__main__":
    main()
