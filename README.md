<div align="center">

<!-- Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,100:48C9B0&height=200&section=header&text=PnP%20Classifier&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Plug-and-Play%20AI%20Classifier&descAlignY=58&descSize=20" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

<br/>

> **A modular, Plug-and-Play MLP classifier pipeline — train once, swap anywhere.**

</div>

---

## 🧩 What is PnP Classifier?

**PnP Classifier** is a lightweight, modular AI classification system built around a **Multi-Layer Perceptron (MLP)** architecture.

It includes a synthetic data **generator**, a full **training notebook**, and a ready-to-use **web application** — making it ideal for rapid experimentation and prototyping.

---

## ✨ Features

- 🔌 **Plug-and-Play design** — modular components, easy to extend or replace
- 🧠 **MLP Classifier** — neural network-based classification via scikit-learn
- 🎲 **Synthetic data generator** — create custom datasets for testing and training
- 🌐 **Interactive web app** — run inference through a clean UI (`app.py`)
- 📓 **Jupyter Notebook** — step-by-step walkthrough of the full ML pipeline
- 📦 **Clean dependencies** — minimal setup with `requirements.txt`

---

## 📁 Project Structure

```
pnp-classifier/
│
├── 📓 mlp_classifier.ipynb   # Full training & evaluation pipeline
├── ⚙️  app.py                 # Web application for interactive inference
├── 🎲 generator.py           # Synthetic dataset generator
├── 📂 artifacts_pnpai/       # Saved model artifacts & outputs
├── 🖼️  assets/               # Visual assets and resources
└── 📋 requirements.txt       # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/tatic2003/pnp-classifier.git
cd pnp-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset and train the model (optional you can use ./artifacts_pnpai)

Open and run the notebook:

```bash
jupyter notebook mlp_classifier.ipynb
```

### 5. Launch the web app

```bash
streamlit run app.py
```

---

## 🧠 Model Overview

The classifier uses a **Multi-Layer Perceptron (MLP)** — a feedforward neural network well-suited for tabular classification tasks.

| Parameter        | Value              |
|------------------|--------------------|
| Model Type       | MLP Classifier     |
| Framework        | scikit-learn       |
| Input            | Feature vectors    |
| Output           | Class predictions  |
| Training         | Supervised learning|

---

## 🔧 How It Works

```
Raw Data / Generated Data
        ↓
  [ generator.py ]  ──── Creates synthetic training samples
        ↓
[ mlp_classifier.ipynb ] ── Preprocessing → Training → Evaluation
        ↓
  [ artifacts_pnpai/ ] ──── Saves trained model & metrics
        ↓
     [ app.py ] ──────────── Loads model → Serves predictions via UI
```

---

## 📊 Notebook Walkthrough

The `mlp_classifier.ipynb` notebook covers:

1. 📥 Data loading and exploration
2. 🔧 Feature preprocessing and normalization
3. 🏋️ MLP model training
4. 📈 Evaluation: accuracy, confusion matrix, classification report
5. 💾 Model serialization for deployment

---

## 📦 Dependencies

Key libraries used in this project:

| Library         | Purpose                        |
|-----------------|--------------------------------|
| `scikit-learn`  | MLP model & metrics            |
| `numpy`         | Numerical computing            |
| `pandas`        | Data manipulation              |
| `matplotlib`    | Visualization                  |
| `gradio` / `streamlit` | Web app interface       |

> See [`requirements.txt`](requirements.txt) for the full list.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Open an issue for bugs or feature requests
- 🍴 Fork the repo and submit a pull request
- ⭐ Star the project if you find it useful!

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:48C9B0,100:6C63FF&height=100&section=footer" width="100%"/>

Made with ❤️ by [tatic2003](https://github.com/tatic2003)

</div>
