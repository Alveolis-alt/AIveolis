
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib.backends.backend_pdf import PdfPages

st.set_option('client.showErrorDetails', True)
st.set_page_config(layout="wide")
st.title("AIveolis - Interpretación de Rayos X con Grad-CAM y LIME")

# Cargar modelo EfficientNetB0
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights=None)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
model.load_weights("modelo_pesos.keras")

try:
    with open("history.json", "r") as f:
        history = json.load(f)
except:
    history = None

# Funciones
def get_img_array(img, size=(224, 224)):
    img = img.resize(size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(array)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        output = predictions[:, pred_index]

    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    return tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap), predictions[0].numpy()

def generate_lime_explanation(model, image):
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        images = tf.image.resize(images, (224, 224)).numpy()
        images = tf.keras.applications.efficientnet.preprocess_input(images)
        return model.predict(images)

    explanation = explainer.explain_instance(
        np.array(image),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    return mark_boundaries(temp / 255.0, mask)

# Formulario paciente
st.sidebar.header("Información del paciente")
nombre = st.sidebar.text_input("Nombre")
edad = st.sidebar.number_input("Edad", min_value=0, max_value=120, value=30)
medico = st.sidebar.text_input("Médico tratante")
sintomas = st.sidebar.text_area("Síntomas", height=100)
observaciones = st.sidebar.text_area("Observaciones", height=100)

# Imagen
uploaded_file = st.file_uploader("Sube una imagen de rayos X", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", width=300)

    img_array = get_img_array(image)
    heatmap, predictions = make_gradcam_heatmap(img_array, model)

    pred_class = np.argmax(predictions)
    class_name = f"Clase predicha: {pred_class}"

    # Grad-CAM
    img = np.array(image.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    gradcam_image = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    st.subheader("Grad-CAM: Zonas activadas por el modelo")
    st.image(gradcam_image, caption="Grad-CAM", width=300)
    st.markdown("- 🔥 El modelo se enfoca en estas regiones para tomar su decisión.")

    # LIME
    st.subheader("LIME: Regiones explicativas de la predicción")
    lime_image_np = generate_lime_explanation(model, image.resize((224, 224)))
    st.image(lime_image_np, caption="LIME", width=300)
    st.markdown("- 🎯 LIME destaca las regiones de la imagen más relevantes para la predicción.")

    # Datos
    st.markdown("---")
    st.subheader("🧾 Datos del paciente")
    st.write(f"**Nombre:** {nombre or 'No especificado'}")
    st.write(f"**Edad:** {edad}")
    st.write(f"**Médico:** {medico or 'No especificado'}")
    st.write(f"**Síntomas:** {sintomas or 'No especificados'}")
    st.write(f"**Observaciones:** {observaciones or 'No especificadas'}")
    st.markdown(f"**Resultado del modelo:** {class_name}")

    # PDF
    if st.button("📄 Generar informe PDF"):
        pdf_path = "informe_paciente.pdf"
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title("Imagen original")
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title("Grad-CAM")
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(lime_image_np)
            ax.axis("off")
            ax.set_title("LIME")
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            text = f"Nombre: {nombre}\nEdad: {edad}\nMédico: {medico}\nSíntomas: {sintomas}\nObservaciones: {observaciones}\nResultado del modelo: {class_name}"
            ax.text(0, 0.5, text, fontsize=12, verticalalignment='center')
            pdf.savefig(fig)
            plt.close()

        st.success("Informe PDF generado correctamente.")
        with open(pdf_path, "rb") as f:
            st.download_button("Descargar informe PDF", f, file_name=pdf_path, mime="application/pdf")

    # Histórico
    if history:
        st.subheader("📈 Entrenamiento del modelo")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history['accuracy'], label='Entrenamiento')
        ax[0].plot(history['val_accuracy'], label='Validación')
        ax[0].set_title('Precisión')
        ax[0].legend()
        ax[1].plot(history['loss'], label='Entrenamiento')
        ax[1].plot(history['val_loss'], label='Validación')
        ax[1].set_title('Pérdida')
        ax[1].legend()
        st.pyplot(fig)
else:
    st.info("Por favor sube una imagen para comenzar el análisis.")
