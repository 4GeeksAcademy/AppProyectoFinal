import streamlit as st

def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox("Selecciona una página:", ["Inicio", "Métricas de Rendimiento", "Curva ROC", "Matriz de Confusión", "Visualización de Filtros", "Predicción en Lote", "Historial de Entrenamiento", "Resumen del Modelo"])

    if page == "Inicio":
        st.title("Bienvenido a la aplicación de visualización de Grad-CAM")
        st.write("Usa la barra lateral para navegar entre las diferentes secciones.")
    elif page == "Métricas de Rendimiento":
        display_metrics()
    elif page == "Curva ROC":
        display_roc_curve()
    elif page == "Matriz de Confusión":
        display_confusion_matrix()
    elif page == "Visualización de Filtros":
        display_filters(load_model('models/BestModel-densenet121_preprocessed-retinal.h5'))
    elif page == "Predicción en Lote":
        batch_prediction()
    elif page == "Historial de Entrenamiento":
        display_training_history('path/to/training_history.csv')
    elif page == "Resumen del Modelo":
        display_model_summary()

if __name__ == "__main__":
    main()
