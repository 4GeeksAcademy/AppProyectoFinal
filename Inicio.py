import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import streamlit as st

def main():
    st.title('Portal predictivo Detección de Retinopatía Diabética' )
    st.image('./images/R.jpg', use_column_width=True)
    st.write('**Por favor seleccione la opcion que desea explorar:**')

    opcion = st.radio('Servicios:',
                      ('Predicción de enfermedad (archivo fotográfico)', 'Métricas del modelo', 'Grad-CAM',
                       'Grad-CAM (carga imagen)', 'Curva ROC', 'Curva ROC (carga csv)', 'ConfusionMatrix'),
                      index=0,
                      key='option')

    if st.button('OK'):
        route_prediction(opcion)

def route_prediction(opcion):
    if opcion == 'Predicción de enfermedad (archivo fotográfico)':
        switch_page("Predict")
    elif opcion == 'Métricas del modelo':
        switch_page("metricas-v.3")
    elif opcion== 'Grad-CAM':
        switch_page("Grad-CAM")
    elif opcion== 'Grad-CAM (carga imagen)':
        switch_page('Grad-CAM (carga imagen)')
    elif opcion== 'Curva ROC':
        switch_page('Curva ROC')
    elif opcion== 'Curva ROC (carga csv)':
        switch_page('Curva ROC (carga csv)')
    elif opcion== 'ConfusionMatrix':
        switch_page('ConfusionMatrix')


if __name__ == "__main__":
    main()
