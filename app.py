from PIL import Image
import io
import streamlit as st
import numpy as np
import torch

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title("🔍 Detección Inteligente de Objetos")
st.caption("📸 Apunta la cámara y descubre qué hay en tu entorno en tiempo real")

with st.spinner("Cargando modelo..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("⚙️ Ajustes")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Máximo de objetos", 10, 2000, 1000, 10)

    picture = st.camera_input("📷 Captura una imagen")

    if picture:
        bytes_data = picture.getvalue()

        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR

        with st.spinner("🔍 Analizando imagen..."):
            results = model(
                np_img,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=int(max_det)
            )

        result    = results[0]
        boxes     = result.boxes
        annotated = result.plot()
        annotated_rgb = annotated[:, :, ::-1]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Imagen analizada")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("🧠 Lo que veo")

            if boxes is not None and len(boxes) > 0:
                label_names = model.names
                category_count = {}

                for box in boxes:
                    cat = int(box.cls.item())
                    category_count[cat] = category_count.get(cat, 0) + 1

                # Diccionario de mensajes bonitos
                mensajes = {
                    "person": "👥 Hay personas en la imagen",
                    "cell phone": "📱 Veo un celular",
                    "bottle": "🥤 Hay una botella",
                    "cup": "☕ Veo una taza",
                    "laptop": "💻 Hay un computador",
                    "book": "📚 Veo un libro",
                    "chair": "🪑 Hay una silla",
                    "tv": "📺 Hay una pantalla",
                }

                for cat, count in category_count.items():
                    nombre = label_names[cat]

                    if nombre in mensajes:
                        texto = mensajes[nombre]
                    else:
                        texto = f"👀 Detecté {nombre}"

                    if count > 1:
                        st.success(f"{texto} (x{count})")
                    else:
                        st.success(texto)

            else:
                st.warning("🤔 No detecté objetos claros en esta imagen")

else:
    st.error("❌ No se pudo cargar el modelo")
    st.stop()

st.markdown("---")
st.caption("✨ App de detección con IA (YOLO + Streamlit)")
