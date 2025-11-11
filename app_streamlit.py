# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para predecir morosidad de clientes
Modelos: Keras y PyTorch comparados visualmente
Autor: Lucero Manrique
"""

# =============================================
# 📦 1. Importar librerías
# =============================================
import streamlit as st
import numpy as np
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =============================================
# 💾 2. Cargar escalador
# =============================================
with open("scaler_credito.pkl", "rb") as f:
    scaler = pickle.load(f)

# =============================================
# 🧠 3. Definir modelo PyTorch
# =============================================
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.output(self.relu(self.layer1(x))))

# =============================================
# 🧩 4. Configurar interfaz Streamlit
# =============================================
st.set_page_config(page_title="Predicción de Morosidad", page_icon="💳", layout="centered")
st.title("💳 Predicción de Morosidad de Clientes")
st.write("Ingrese los datos del cliente y seleccione el modelo para la predicción.")

# =============================================
# 🗂 Selector de modelo
# =============================================
modelo_opcion = st.selectbox("Selecciona el modelo a usar:", ("Keras", "PyTorch", "Ambos"))

# =============================================
# 🧾 Entradas del usuario
# =============================================
col1, col2 = st.columns(2)

with col1:
    LIMIT_BAL = st.number_input("Monto de crédito otorgado (NT$)", min_value=0, value=20000, step=1000)
    SEX = st.selectbox("Sexo", options=[1, 2], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
    EDUCATION = st.selectbox("Nivel educativo", options=[1, 2, 3, 4], format_func=lambda x: {
        1: "Postgrado", 2: "Universidad", 3: "Secundaria", 4: "Otros"
    }[x])
    MARRIAGE = st.selectbox("Estado civil", options=[1, 2, 3], format_func=lambda x: {
        1: "Casado", 2: "Soltero", 3: "Otros"
    }[x])
    AGE = st.slider("Edad (años)", 18, 80, 35)

with col2:
    PAY_0 = st.number_input("Estado de pago mes 1 (Sept 2005)", min_value=-1, max_value=9, value=0)
    PAY_2 = st.number_input("Estado de pago mes 2 (Ago 2005)", min_value=-1, max_value=9, value=0)
    PAY_3 = st.number_input("Estado de pago mes 3 (Jul 2005)", min_value=-1, max_value=9, value=0)
    PAY_4 = st.number_input("Estado de pago mes 4 (Jun 2005)", min_value=-1, max_value=9, value=0)
    PAY_5 = st.number_input("Estado de pago mes 5 (May 2005)", min_value=-1, max_value=9, value=0)
    PAY_6 = st.number_input("Estado de pago mes 6 (Abr 2005)", min_value=-1, max_value=9, value=0)

st.write("### 💰 Montos de facturas y pagos anteriores (en NT$)")
col_bill, col_pay = st.columns(2)
bill_values = [col_bill.number_input(f"Factura mes {i} (BILL_AMT{i})", min_value=0, value=5000*i, step=1000) for i in range(1, 7)]
pay_values = [col_pay.number_input(f"Pago mes {i} (PAY_AMT{i})", min_value=0, value=2000*i, step=500) for i in range(1, 7)]

# =============================================
# 🧮 Construir vector de entrada
# =============================================
input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                        PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                        *bill_values, *pay_values]])
input_scaled = scaler.transform(input_data)

# 🔮 Predicción y visualización
# --- Botón de predicción ---
if st.button("🔍 Predecir probabilidad de morosidad"):

    resultados = {}

    # --- Keras ---
    if modelo_opcion in ["Keras", "Ambos"]:
        model_keras = load_model("modelo_credito_simple.h5")
        prob_keras = float(model_keras.predict(input_scaled)[0][0])
        resultados["Keras"] = prob_keras

    # --- PyTorch ---
    if modelo_opcion in ["PyTorch", "Ambos"]:
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        input_dim = input_scaled.shape[1]
        model_pytorch = SimpleModel(input_dim)
        model_pytorch.load_state_dict(torch.load("modelo_pytorch_credito.pth", map_location=torch.device('cpu')))
        model_pytorch.eval()
        with torch.no_grad():
            prob_pytorch = float(model_pytorch(input_tensor).item())
        resultados["PyTorch"] = prob_pytorch

    # --- Mostrar resultados individuales ---
    for nombre, prob in resultados.items():
        estado = "Moroso" if prob > 0.5 else "No Moroso"
        color = "red" if prob > 0.5 else "green"
        st.markdown(f"### {nombre}: <span style='color:{color}'>{prob*100:.2f}% - {estado}</span>", unsafe_allow_html=True)
        st.progress(prob)

    # --- Guardar resultados en session_state para usar después ---
    st.session_state["resultados"] = resultados

# --- Checkbox para mostrar gráfico comparativo ---
if modelo_opcion == "Ambos" and "resultados" in st.session_state:
    mostrar_grafico = st.checkbox("Mostrar gráfico comparativo")
    if mostrar_grafico:
        resultados = st.session_state["resultados"]
        modelos = list(resultados.keys())
        probs = [resultados[m]*100 for m in modelos]
        colores = ["green" if p <= 50 else "red" for p in probs]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(modelos, probs, color=colores)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probabilidad de morosidad (%)')
        ax.set_title('Comparación de predicciones por modelo')

        # Etiquetas encima de las barras
        for bar in bars:
            width = bar.get_width()
            estado = "Moroso" if width > 50 else "No Moroso"
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}% - {estado}', va='center', fontsize=12)

        st.pyplot(fig)
        plt.close(fig)



