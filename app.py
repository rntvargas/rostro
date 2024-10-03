import cv2  # Importando la biblioteca OpenCV para tareas de visión por computadora
import streamlit as st  # Importando Streamlit para construir aplicaciones web interactivas
import numpy as np  # Importando NumPy para cálculos numéricos
from PIL import Image  # Importando la Biblioteca de Imágenes de Python para el procesamiento de imágenes

# Función para detectar caras y ojos en una imagen subida
def detectar_caras_y_ojos_en_imagen(imagen_subida):
    # Convertir el archivo de imagen subido a un array de NumPy
    img_array = np.array(Image.open(imagen_subida))

    # Crear el cascada haar para la detección de caras y ojos usando archivos XML preentrenados
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # Convertir la imagen a escala de grises para la detección de caras y ojos
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detectar caras en la imagen en escala de grises
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # Parámetro que especifica cuánto se reduce el tamaño de la imagen en cada escala de imagen
        minNeighbors=10,  # Parámetro que especifica cuántos vecinos debe tener cada rectángulo candidato para retenerlo
        minSize=(30, 30)  # Tamaño mínimo posible del objeto. Los objetos más pequeños que esto serán ignorados
    )

    # Dibujar un rectángulo alrededor de cada cara detectada y detectar ojos dentro de la cara
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_array[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    # Mostrar la imagen resultante con detección de caras y ojos
    st.image(img_array, channels="RGB", use_column_width=True)

# Interfaz de usuario de Streamlit
st.title("Detección de Caras y Ojos")
st.subheader("Sube una imagen para detectar caras y ojos")

# Cargador de archivos para detectar caras y ojos en una imagen subida
imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if imagen_subida is not None:
    detectar_caras_y_ojos_en_imagen(imagen_subida)

