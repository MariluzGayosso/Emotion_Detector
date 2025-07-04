import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Cargar modelo
modelo = load_model('modelos/modelo_emociones.hdf5', compile=False)

# Etiquetas del dataset FER-2013
emociones = ['Enojo', 'Disgusto', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

# Clasificador de rostro
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Estilos personalizados
st.markdown("""
    <style>
        .title {
            font-size: 46px;
            text-align: center;
            color: color: #b93e63;;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .emojis {
            font-size: 38px;
            color: #444;
            text-align: center;
            margin-bottom: 20px;
        } 
        .subtext {
            font-size: 18px;
            color: #444;
            text-align: center;
            margin-bottom: 30px;
        }
        .resultado {
            background-color: #212f3c;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 30px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
        .resultado p {
            color: #f1f1f3;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="title">- Detector de Emociones con IA -</div>', unsafe_allow_html=True)
st.markdown('<div class="emojis">🧠😃😭😡</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Sube una imagen y descubre qué emoción refleja tu rostro</div>', unsafe_allow_html=True)

# Cargar imagen
imagen_cargada = st.file_uploader("📤 Sube una imagen con rostro", type=["jpg", "jpeg", "png"])

if imagen_cargada is not None:
    # Imagen original
    img = Image.open(imagen_cargada)
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Copias para mostrar
    img_original = img_np.copy()
    img_detectada = img_np.copy()

    # Detectar rostro
    caras = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(caras) == 0:
        st.warning("⚠️ No se detectó ningún rostro. Intenta con otra imagen.")
    else:
        for (x, y, w, h) in caras:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (64, 64))
            rostro = rostro.astype("float32") / 255.0
            rostro = np.expand_dims(rostro, axis=0)
            rostro = np.expand_dims(rostro, axis=-1)

            prediccion = modelo.predict(rostro)
            emocion_index = np.argmax(prediccion)
            emocion = emociones[emocion_index]
            porcentaje = float(prediccion[0][emocion_index]) * 100

            # Dibujar sobre imagen procesada
            cv2.rectangle(img_detectada, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img_detectada, emocion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostrar ambas imágenes lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, caption="📷 Imagen original", use_column_width=True)
        with col2:
            st.image(img_detectada, caption="🧍 Rostro detectado y emoción marcada", use_column_width=True)

        # Mostrar resultados abajo
        st.markdown(f"""
            <div class="resultado">
                <p>😄  Emoción detectada : <strong>{emocion}</strong></p>
                <p>📊  Porcentaje de certeza : <strong>{porcentaje:.2f}%</strong></p>
            </div>
        """, unsafe_allow_html=True)

