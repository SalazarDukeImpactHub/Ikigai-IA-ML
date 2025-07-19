# streamlit_app.py (Versión Final con Mejoras Funcionales y de Contenido)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import time

# --- INYECCIÓN DE CSS PARA MEJORAR LA INTERFAZ ---
def remote_css(css_string):
    """Función para inyectar CSS personalizado en la app."""
    st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)

# CSS personalizado para un diseño más moderno y profesional
custom_css = """
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
.result-card {
    background-color: white;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border: 1px solid #e6e6e6;
    transition: transform 0.2s;
}
.result-card:hover {
    transform: scale(1.02);
}
div.stButton > button:first-child {
    background-color: #4B8BBE;
    color: white;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    width: 100%;
}
div.stButton > button:first-child:hover {
    background-color: #3A6A94;
    border: none;
}
h4 {
    color: #1E3A5F;
    font-weight: bold;
}
"""
remote_css(custom_css)

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Ikigai-ML: Tu Orientador Profesional",
    page_icon="🤖",
    layout="wide"
)

# --- Carga de Datos y Modelos ---
@st.cache_data
def cargar_activos():
    """Carga todos los archivos necesarios para la aplicación una sola vez."""
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "data"
    try:
        knn_model = joblib.load(data_path / "knn_model.pkl")
        pivot_onet = pd.read_parquet(data_path / "mat_full.parquet")
        df_onet_titulos = pd.read_parquet(data_path / "onet_titles.parquet")
        df_puente = pd.read_parquet(data_path / "puente_onet_dane_ia.parquet")
        df_dane = pd.read_parquet(data_path / "dane_enriquecido_final_2024.parquet")
        df_traducciones = pd.read_parquet(data_path / "habilidades_traduccion.parquet")
    except FileNotFoundError as e:
        st.error(f"Error Crítico al Cargar Archivo: {e}. La aplicación no puede iniciar.")
        st.info("Asegúrate de que la estructura de tu repositorio sea 'src/data/' y que todos los archivos estén allí.")
        return None
    return knn_model, pivot_onet, df_onet_titulos, df_puente, df_dane, df_traducciones

activos = cargar_activos()

# --- Función de Vectorización ---
def vector_usuario(user_skills, reference_matrix):
    """Convierte una lista de habilidades en un vector numérico para el modelo."""
    vector = np.zeros((1, len(reference_matrix.columns)))
    skills_encontradas = [skill for skill in user_skills if skill in reference_matrix.columns]
    if skills_encontradas:
        for skill in skills_encontradas:
            vector[0, reference_matrix.columns.get_loc(skill)] = 1
        return vector / vector.sum()
    return vector

# --- NAVEGACIÓN EN LA BARRA LATERAL ---
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Elige una sección", ["Orientador Profesional", "Acerca del Proyecto"])

# --- PÁGINA PRINCIPAL: ORIENTADOR PROFESIONAL ---
if pagina == "Orientador Profesional":
    st.title("🤖 Proyecto Ikigai-ML")
    st.header("Tu Orientador Profesional con Inteligencia Artificial")

    # --- NUEVO: SECCIÓN EXPLICATIVA (METODOLOGÍA) ---
    with st.expander("🤔 ¿Cómo funciona esta herramienta?"):
        st.write("""
            Esta aplicación utiliza un modelo de Inteligencia Artificial para ayudarte a descubrir tu vocación. El proceso es el siguiente:
            1.  **Datos de O\*NET:** Usamos la base de datos O\*NET del Departamento de Trabajo de EE.UU., que describe cientos de profesiones según las habilidades que requieren.
            2.  **Datos del DANE:** Cruzamos esta información con la Gran Encuesta Integrada de Hogares del DANE (2024) de Colombia para entender qué tan comunes son estas profesiones en nuestro país.
            3.  **Inteligencia Artificial:** Un modelo de IA encuentra las profesiones internacionales que mejor se corresponden con las ocupaciones locales.
            4.  **Tu Perfil:** Al seleccionar tus habilidades, nuestro recomendador las compara con las de todas las profesiones y te muestra las 5 más compatibles contigo.
        """)

    st.markdown("---")

    if activos:
        (knn_model, pivot_onet, df_onet_titulos, df_puente, df_dane, df_traducciones) = activos
        
        st.subheader("Paso 1: Selecciona tus Habilidades")
        mapa_es_a_en = pd.Series(df_traducciones.skill_en.values, index=df_traducciones.skill_es).to_dict()
        opciones_habilidades_es = sorted(mapa_es_a_en.keys())
        
        habilidades_seleccionadas_es = st.multiselect(
            "Selecciona tus habilidades de la lista (puedes escribir para buscar):",
            options=opciones_habilidades_es,
            placeholder="Elige una o varias habilidades"
        )
        st.markdown("---")
        
        if st.button("Encontrar mi Ikigai ✨"):
            if habilidades_seleccionadas_es:
                with st.spinner('Analizando tu perfil y buscando las mejores profesiones...'):
                    time.sleep(1.5) # Simula un retraso para que el spinner sea visible
                    habilidades_en_ingles = [mapa_es_a_en[skill_es] for skill_es in habilidades_seleccionadas_es]
                    
                    st.subheader("Paso 2: Tus Profesiones Recomendadas")
                    
                    u_vec = vector_usuario(habilidades_en_ingles, pivot_onet)
                    distances, indices = knn_model.kneighbors(u_vec, n_neighbors=5)
                    
                    onet_results = df_onet_titulos.iloc[indices[0]]
                    
                    # --- NUEVO: Lista para guardar los datos del gráfico ---
                    resultados_grafico = []

                    for onet_soc_code, row in onet_results.iterrows():
                        titulo_onet = row['Title']
                        
                        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                        st.markdown(f"#### 💼 {titulo_onet}")

                        info_local = df_puente[df_puente['Onet_Title'] == titulo_onet]
                        
                        if not info_local.empty:
                            nombre_dane = info_local['Dane_Name'].iloc[0]
                            descripcion_dane = info_local['Dane_Description'].iloc[0]
                            similitud = info_local['Similarity_Score'].iloc[0]
                            conteo = len(df_dane[df_dane['Nombre Ocupación'] == nombre_dane])

                            st.info(f"**Ocupación Equivalente en Colombia (IA):** {nombre_dane}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="🎯 Afinidad de Significado", value=f"{similitud:.0%}")
                            with col2:
                                st.metric(label="📊 Presencia en Encuesta DANE 2024", value=f"{conteo:,}".replace(',', '.'))
                            
                            with st.expander("Ver Descripción del Perfil en Colombia (DANE)"):
                                st.write(descripcion_dane)
                            
                            # Guardamos los datos para el gráfico
                            resultados_grafico.append({'Profesión en Colombia': nombre_dane, 'Presencia en DANE': conteo})
                        else:
                            st.warning("No se encontró una equivalencia semántica directa en los datos de Colombia para esta profesión.")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # --- NUEVO: GRÁFICO COMPARATIVO DE RESULTADOS ---
                    if resultados_grafico:
                        st.markdown("---")
                        st.subheader("Comparativa de Popularidad en Colombia")
                        
                        df_grafico = pd.DataFrame(resultados_grafico)
                        df_grafico = df_grafico.set_index('Profesión en Colombia')
                        
                        st.bar_chart(df_grafico['Presencia en DANE'])
                        st.caption("Este gráfico muestra cuántas veces apareció cada profesión recomendada en la encuesta del DANE, indicando su popularidad relativa en el mercado laboral colombiano.")

            else:
                st.warning("Por favor, selecciona al menos una habilidad de la lista.")
    else:
        st.error("La aplicación no pudo iniciar porque faltan archivos de datos esenciales.")

# --- PÁGINA "ACERCA DE" ---
elif pagina == "Acerca del Proyecto":
    st.title("💡 Acerca de Ikigai-ML")
    st.markdown("---")
    
    st.header("Nuestra Misión")
    st.write(
        """
        El **Proyecto Ikigai-ML** nace de la convicción de que la tecnología y la ciencia de datos pueden ser herramientas poderosas para el desarrollo personal y profesional. 
        Nuestra misión es ofrecer una guía vocacional accesible, inteligente y contextualizada a la realidad del mercado laboral colombiano, ayudando a las personas a encontrar su **Ikigai**: esa intersección entre lo que aman, en lo que son buenos, lo que el mundo necesita y por lo que pueden ser pagados.
        """
    )

    st.header("La Creadora")
    st.write(
        """
        Este proyecto fue desarrollado por **Jennifer Salazar Duke** como parte de su formación y compromiso con el **Salazar Duke Impact Hub**. 
        El Hub es una iniciativa dedicada a impulsar proyectos con impacto social a través de la tecnología, la educación y la colaboración comunitaria.
        """
    )

    st.header("Tecnología y Datos")
    st.write(
        """
        - **Modelo de Recomendación:** Utilizamos un algoritmo de *Vecinos más Cercanos (k-NN)* para comparar tu perfil de habilidades con cientos de profesiones.
        - **Procesamiento de Lenguaje Natural (NLP):** Un modelo de IA (Sentence Transformers) nos permite encontrar las ocupaciones colombianas que son semánticamente más parecidas a los estándares internacionales.
        - **Fuentes de Datos:**
            - **O\*NET OnLine:** Para los perfiles de habilidades de profesiones.
            - **DANE (Colombia):** Para los datos de frecuencia y descripciones de ocupaciones en Colombia (GEIH 2024).
        """
    )
