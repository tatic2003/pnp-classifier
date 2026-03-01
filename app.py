import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd

#Configuración general
st.set_page_config(page_title="PnP-AI Device Classifier", layout="centered")

st.title("PnP-AI - Clasificador ITS")
st.markdown("Simulación de inferencia Plug-and-Play. " )
st.markdown("Ingrese las características de un dispositivo IoT para clasificar su tipo.")
st.divider()
import tensorflow as tf
import keras
import streamlit as st

st.write("TensorFlow:", tf.__version__)
st.write("Keras:", keras.__version__)
# Cargar modelos
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:/Users/tatic/ドキュメント/MAESTRIA 2025 - FIUNA/TERCER TRIMESTRE/final_demo_Redes_Neuronales/artifacts_pnpai/mlp_classifier.keras")
    scaler = joblib.load("C:/Users/tatic/ドキュメント/MAESTRIA 2025 - FIUNA/TERCER TRIMESTRE/final_demo_Redes_Neuronales/artifacts_pnpai/scaler.joblib")
    encoder = joblib.load("C:/Users/tatic/ドキュメント/MAESTRIA 2025 - FIUNA/TERCER TRIMESTRE/final_demo_Redes_Neuronales/artifacts_pnpai/label_encoder.joblib")
    return model, scaler, encoder
model, scaler, le = load_model()
st.info("Modelo activo: MLP Neural Network")
st.divider()

#Templates de dispositivos para autocompletar campos
DEVICE_TEMPLATES = {
    "Manual": None,
    "ESP32-WROOM32 (WiFi)": {
        "memory_total_mb": 0.5,          # 520 KB ≈ 0.5 MB SRAM
        "memory_free_pct": 0.80,
        "cpu_freq_mhz": 240.0,
        "is_wifi": 1,
        "is_ethernet": 0
    },
    "ESP32 Olimex PoE (Ethernet)": {
        "memory_total_mb": 4.0,          # 4 MB Flash
        "memory_free_pct": 0.75,
        "cpu_freq_mhz": 240.0,
        "is_wifi": 0,
        "is_ethernet": 1
    },
    "Raspberry Pi 2B": {
        "memory_total_mb": 1024.0,       # 1 GB RAM
        "memory_free_pct": 0.60,
        "cpu_freq_mhz": 900.0,
        "is_wifi": 0,
        "is_ethernet": 1
    },
    "Jetson Nano": {
        "memory_total_mb": 4096.0,
        "memory_free_pct": 0.7,
        "cpu_freq_mhz": 1800.0,
        "is_wifi": 1,
        "is_ethernet": 1
    },
    "Gateway Industrial": {
        "memory_total_mb": 4096.0,
        "memory_free_pct": 0.65,
        "cpu_freq_mhz": 2000.0,
        "is_wifi": 0,
        "is_ethernet": 1
    },
    "Sensor simple": {
        "memory_total_mb": 0.1,
        "memory_free_pct": 0.9,
        "cpu_freq_mhz": 70.0,
        "is_wifi": 1,
        "is_ethernet": 0
    }
}

#Botones con imágenes
st.markdown("### Seleccionar perfil rápido")

if "selected_template" not in st.session_state:
    st.session_state.selected_template = "Manual"

col1, col2, col3 = st.columns(3)

with col1:
    st.image("assets/manual.png", width=100)
    if st.button("Manual"):
        st.session_state.selected_template = "Manual"
        
with col2:
    st.image("assets/esp32.png", width=100)
    if st.button("ESP32-WROOM32 (WiFi)"):
        st.session_state.selected_template = "ESP32-WROOM32 (WiFi)"

with col3:
    st.image("assets/rpi.jpg", width=100)
    if st.button("Raspberry Pi 2B"):
        st.session_state.selected_template = "Raspberry Pi 2B"

col4, col5, col6 = st.columns(3)

with col4:
    st.image("assets/jetson.jpg", width=100)
    if st.button("Jetson"):
        st.session_state.selected_template = "Jetson Nano"

with col5:
    st.image("assets/gateway.png", width=100)
    if st.button("Gateway"):
        st.session_state.selected_template = "Gateway Industrial"
with col6:
    st.image("assets/sensor.jpg", width=150)
    if st.button("Sensor"):
        st.session_state.selected_template = "Sensor simple"

col7, col8, col9 = st.columns(3)

with col7:
    st.image("assets/esp32.png", width=100)
    if st.button("ESP32 Olimex PoE"):
        st.session_state.selected_template = "ESP32 Olimex PoE (Ethernet)"

st.divider()
# Aplicar template seleccionado
selected_template = st.session_state.selected_template
template = DEVICE_TEMPLATES[selected_template]

if template:
    memory_total_default = float(template["memory_total_mb"])
    memory_free_default = float(template["memory_free_pct"])
    cpu_freq_default = float(template["cpu_freq_mhz"])
    wifi_default = bool(template["is_wifi"])
    ethernet_default = bool(template["is_ethernet"])
else:
    memory_total_default = 1024.0
    memory_free_default = 0.7
    cpu_freq_default = 1000.0
    wifi_default = False
    ethernet_default = False

# Inputs del usuario
memory_total = st.number_input(
    "Memoria total (MB)",
    min_value=0.01,
    max_value=32000.0,
    value=float(memory_total_default)
)

memory_free_pct = st.slider(
    "Porcentaje de memoria libre",
    min_value=0.0,
    max_value=1.0,
    value=float(memory_free_default),
    step=0.01
)

cpu_freq = st.number_input(
    "Frecuencia CPU (MHz)",
    min_value=1.0,
    max_value=5000.0,
    value=float(cpu_freq_default)
)

is_wifi = st.checkbox("WiFi disponible", value=wifi_default)
is_ethernet = st.checkbox("Ethernet disponible", value=ethernet_default)

st.divider()

# Predicción
if st.button("Clasificar dispositivo"):

    input_data = pd.DataFrame([{
        "memory_total_mb": float(memory_total),
        "memory_free_pct": float(memory_free_pct),
        "cpu_freq_mhz": float(cpu_freq),
        "is_wifi": int(is_wifi),
        "is_ethernet": int(is_ethernet)
    }])

    # Escalar datos
    input_scaled = scaler.transform(input_data)

    # Predicción
    predictions = model.predict(input_scaled)
    pred_class = np.argmax(predictions, axis=1)

    pred_label = le.inverse_transform(pred_class)[0]

    st.success(f"Tipo detectado: {pred_label}")

    # Mostrar probabilidades
    confidence = np.max(predictions) * 100
    st.info(f"Confianza del modelo: {confidence:.2f}%")

    prob_df = pd.DataFrame({
        "Clase": le.classes_,
        "Probabilidad (%)": predictions[0] * 100
    }).sort_values("Probabilidad (%)", ascending=False)

    st.bar_chart(prob_df.set_index("Clase"))