# api.py (Versión Final con Validación de Entrada Robusta)
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import unicodedata

# --- 1. Inicializar la aplicación Flask ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# --- 2. Función de Normalización de Texto ---
def normalizar_texto(texto):
    """Convierte a minúsculas y elimina tildes/acentos."""
    if not isinstance(texto, str):
        return ""
    
    texto = texto.lower().strip()
    nfkd_form = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- 3. Cargar los modelos y datos ---
try:
    base_dir = Path(__file__).resolve().parent
    models_path = base_dir / "models"

    print("Cargando modelos y datos desde la carpeta 'models'...")
    knn_model = joblib.load(models_path / "knn_model.pkl")
    pivot_onet = pd.read_parquet(models_path / "mat_full.parquet")
    df_onet_titulos = pd.read_parquet(models_path / "onet_titles.parquet")
    df_puente = pd.read_parquet(models_path / "puente_onet_dane_ia.parquet")
    df_dane = pd.read_parquet(models_path / "dane_enriquecido_final_2024.parquet")
    df_traducciones = pd.read_parquet(models_path / "habilidades_traduccion.parquet")
    
    mapa_es_a_en = pd.Series(
        df_traducciones.skill_en.values, 
        index=df_traducciones.skill_es.apply(normalizar_texto)
    ).to_dict()
    
    print("✓ Modelos y datos cargados exitosamente.")
    
except Exception as e:
    print(f"Error fatal al cargar los modelos: {e}")
    knn_model = None

# --- 4. Definir las funciones de lógica ---
def vector_usuario(user_skills, reference_matrix):
    """Convierte una lista de habilidades en un vector numérico para el modelo."""
    vector = np.zeros((1, len(reference_matrix.columns)))
    skills_encontradas = [skill for skill in user_skills if skill in reference_matrix.columns]
    if skills_encontradas:
        for skill in skills_encontradas:
            vector[0, reference_matrix.columns.get_loc(skill)] = 1
        return vector / vector.sum()
    return vector

def obtener_recomendaciones_detalladas(habilidades_brutas):
    """Función principal que limpia la entrada y devuelve detalles de profesiones."""
    if knn_model is None:
        return [{"error": "El modelo de recomendación no está disponible."}]

    # --- CORRECCIÓN CLAVE ---
    # Lógica para limpiar y aplanar la lista de entrada, manejando strings, listas y dicts.
    habilidades_limpias = []
    for item in habilidades_brutas:
        if isinstance(item, str):
            habilidades_limpias.append(item)
        elif isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, str):
                    habilidades_limpias.append(sub_item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    habilidades_limpias.append(value)
    
    habilidades_es = list(set(habilidades_limpias)) # Eliminar duplicados

    habilidades_en = [mapa_es_a_en.get(normalizar_texto(h), None) for h in habilidades_es]
    habilidades_en_validas = [h for h in habilidades_en if h is not None]

    if not habilidades_en_validas:
        return [{"error": "No se reconocieron las habilidades proporcionadas."}]

    u_vec_numpy = vector_usuario(habilidades_en_validas, pivot_onet)
    u_vec_df = pd.DataFrame(u_vec_numpy, columns=pivot_onet.columns)
    
    distances, indices = knn_model.kneighbors(u_vec_df, n_neighbors=5)
    
    onet_results = df_onet_titulos.iloc[indices[0]]
    
    resultados_completos = []
    for onet_soc_code, row in onet_results.iterrows():
        titulo_onet = row['Title']
        info_local = df_puente[df_puente['Onet_Title'] == titulo_onet]
        profesion_info = {"profesion_onet": titulo_onet}
        
        if not info_local.empty:
            afinidad_valor = float(info_local['Similarity_Score'].iloc[0])
            presencia_valor = int(len(df_dane[df_dane['Nombre Ocupación'] == info_local['Dane_Name'].iloc[0]]))

            profesion_info["info_colombia"] = {
                "nombre_dane": info_local['Dane_Name'].iloc[0],
                "descripcion": info_local['Dane_Description'].iloc[0],
                "afinidad": round(afinidad_valor, 2),
                "presencia_dane": presencia_valor
            }
        
        resultados_completos.append(profesion_info)
        
    return resultados_completos

# --- 5. Crear el endpoint de la API ---
@app.route('/recomendar', methods=['POST'])
def recomendar():
    """Endpoint para recibir habilidades y devolver profesiones con detalles."""
    try:
        datos = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Error al decodificar JSON: {e}"}), 400

    if not datos or 'habilidades' not in datos or not isinstance(datos['habilidades'], list):
        return jsonify({"error": "La petición debe ser un JSON con una lista en la clave 'habilidades'."}), 400
        
    habilidades_usuario = datos['habilidades']
    profesiones_detalladas = obtener_recomendaciones_detalladas(habilidades_usuario)
    
    return jsonify({"recomendaciones": profesiones_detalladas})

# --- 6. Iniciar el servidor Flask ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)



