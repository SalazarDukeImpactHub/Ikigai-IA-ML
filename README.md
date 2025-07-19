# Ikigai IA – Orientador Vocacional Inteligente

Ikigai IA es un **orientador vocacional inteligente** que usa *machine learning* y *procesamiento de lenguaje natural (NLP)* para ayudar a personas en Colombia a encontrar profesiones que se alineen con sus talentos, intereses y la demanda del mercado laboral.

El proyecto integra datos de **O*NET** (EE.UU.) y la **Gran Encuesta Integrada de Hogares (GEIH) del DANE** (Colombia), conectados mediante un modelo multilingüe de *sentence-transformers*.  
Está desplegado como aplicación web con **Flask** en Hugging Face Spaces y como **chatbot interactivo en Telegram**, gracias a la integración con **Make IA** y **ChatGPT**.

---

## Accesos Rápidos

- **Presentación interactiva (con videos):**  
  [Ver en Canva](https://www.canva.com/design/DAGtnSawH0U/NJgsoV12p8wCuanNwrZ1-Q/view?utm_content=DAGtnSawH0U&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h49a32ea943)

- **Aplicación Web (Hugging Face):**  
  [Probar Ikigai IA](https://huggingface.co/spaces/jennifersalazarduke/ikigai-ia)

- **Chatbot en Telegram:**  
  [Interactuar con Ikigai IA](https://t.me/IkigaiML_bot)

---

## Instalación Local

1. Clonar el repositorio:

```bash
git clone https://github.com/SalazarDukeImpactHub/Ikigai-IA-ML.git
cd Ikigai-IA-ML
```

2. Crear un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación localmente:

```bash
flask run
```

La app se ejecutará en `http://127.0.0.1:5000/`.

---

## Estructura del Proyecto

```
Ikigai-IA-ML/
│
├── app.py                      # Aplicación Flask
├── Proyecto_Ikigai_ml.ipynb    # Notebook de desarrollo y pruebas
├── puente_onet_dane_ia.parquet # Archivo de correspondencias semánticas
├── requirements.txt            # Dependencias
├── README.md                   # Documentación
│
├── static/                     # Archivos estáticos (CSS, JS, imágenes)
├── templates/                  # Vistas HTML de Flask
├── data/                       # Datos locales (ejemplo: GEIH)
└── scripts/                    # Scripts de procesamiento de datos
```

---

## Tecnologías Principales

- **Flask** (backend y API)
- **scikit-learn** (modelo k-NN)
- **sentence-transformers** (NLP multilingüe)
- **pandas, numpy** (procesamiento de datos)
- **Make IA + Telegram Bot** (interacción automatizada con usuarios)

---

## Licencia

Este proyecto está bajo licencia MIT. Puedes usarlo, modificarlo y distribuirlo libremente con atribución.
