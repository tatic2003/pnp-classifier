# PnP-AI Device Classifier

A lightweight **IoT device type classifier** for Intelligent Transportation Systems (ITS), built with a neural network and a simple Streamlit interface.

This repository contains:
- A Streamlit app for interactive inference.
- A synthetic dataset generator for IoT/ITS devices.
- Trained artifacts (`.keras`, scaler, and label encoder) for quick local demos.

---

## ✨ What this project does

Given a small set of device characteristics (memory, free memory ratio, CPU frequency, and network interfaces), the app predicts the most likely device class (for example: ESP32-based node, sensor, gateway, SBC).

The goal is to simulate a **Plug-and-Play onboarding flow** where unknown devices can be automatically profiled.

---

## 🧱 Repository layout

```text
.
├── app.py                     # Streamlit inference UI
├── generator.py               # Synthetic dataset generator
├── requirements.txt           # Python dependencies
├── mlp_classifier.ipynb       # Notebook used for experimentation/training
├── artifacts_pnpai/
│   ├── mlp_classifier.keras   # Trained neural network model
│   ├── scaler.joblib          # Feature scaler
│   └── label_encoder.joblib   # Output class encoder
└── assets/                    # UI images used by Streamlit app
```

---

## 🚀 Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

Open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## 🧪 Using the classifier

1. Open the app.
2. Choose a quick profile (ESP32, Raspberry Pi, Jetson, Gateway, Sensor) or use Manual mode.
3. Adjust hardware/network feature values.
4. Click **“Clasificar dispositivo”** to get:
   - Predicted class.
   - Confidence score.
   - Probability chart by class.

---

## 🛠 Synthetic data generation

`generator.py` can generate realistic IoT/ITS datasets for training and testing.

### Example: generate data for classification

```python
from generator import create_generator

gen = create_generator(seed=42)
features, labels = gen.generate_classification_dataset(
    num_samples=1000,
    include_anomalies=True,
    anomaly_rate=0.1
)

print(len(features), len(labels))
print(features[0], labels[0])
```

### Example: export raw device records

```python
from generator import create_generator

gen = create_generator(seed=42)
devices = gen.generate_dataset(num_devices=500)
gen.save_dataset(devices, "data/devices.json", format="json")
gen.save_dataset(devices, "data/devices.csv", format="csv")
```

---

## ⚠️ Important note about model paths

The current `app.py` loads artifacts from **absolute Windows paths**. If you run this project in another environment, update `load_model()` to use repository-relative paths such as:

- `artifacts_pnpai/mlp_classifier.keras`
- `artifacts_pnpai/scaler.joblib`
- `artifacts_pnpai/label_encoder.joblib`

---

## 📦 Dependencies

Main libraries:
- `streamlit`
- `tensorflow` / `keras`
- `pandas`
- `numpy`
- `joblib`

See `requirements.txt` for the exact pinned versions.

---

## 🔭 Roadmap ideas

- Replace absolute artifact paths with robust relative-path loading.
- Add model version metadata to the UI.
- Provide a training script (`train.py`) for reproducible retraining.
- Add tests for feature preprocessing and inference pipeline.
- Add Docker support for one-command deployment.

---

## 🤝 Acknowledgment

Based on ITS-oriented synthetic profiling work described in the project comments (`PINV01-24 - FIUNA`).
