"""
exportar_dashboard.py
=====================

Versión coherente con la Sección 4 del notebook.

Predice ap_hi(mmHg) usando las cinco variables con mayor correlación:
- ap_lo(mmHg)
- cardio
- weight(kg)
- age
- cholesterol

Se conserva el orden de entrada usado en el notebook:
age, weight(kg), ap_lo(mmHg), cholesterol, cardio
"""

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


SEMILLA = 42
TEST_SIZE = 0.2

ruta_archivo = (
    "/content/drive/MyDrive/AnDa_ProyectoFinal_Data/cardio_train.csv"
)

df_cardios = pd.read_csv(ruta_archivo, sep=";")

mask_pressure = (
    (df_cardios["ap_hi"] > df_cardios["ap_lo"])
    & (df_cardios["ap_hi"] < 250)
    & (df_cardios["ap_hi"] > 60)
    & (df_cardios["ap_lo"] < 150)
    & (df_cardios["ap_lo"] > 40)
)
mask_height = (
    (df_cardios["height"] >= 100)
    & (df_cardios["height"] <= 200)
)
mask_weight = (
    (df_cardios["weight"] >= 40)
    & (df_cardios["weight"] <= 200)
)

df_cardiosclean = df_cardios[
    mask_pressure & mask_height & mask_weight
].copy()

df_cardiosclean["age"] = (
    df_cardiosclean["age"] / 365.25
).astype(int)

df_cardiosclean.drop_duplicates(inplace=True)

df_cardiosclean.rename(
    columns={
        "height": "height(cm)",
        "weight": "weight(kg)",
        "ap_hi": "ap_hi(mmHg)",
        "ap_lo": "ap_lo(mmHg)",
    },
    inplace=True,
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

df_cardiosclean.to_csv(
    "data/cardio_clean.csv",
    index=False,
)

# Variables de mayor correlación con ap_hi en el análisis del notebook.
# Orden conservado de la Sección 4.
FEATURES_REG = [
    "age",
    "weight(kg)",
    "ap_lo(mmHg)",
    "cholesterol",
    "cardio",
]
TARGET_REG = "ap_hi(mmHg)"

Xr = df_cardiosclean[FEATURES_REG]
yr = df_cardiosclean[TARGET_REG]

xr_train, xr_test, yr_train, yr_test = train_test_split(
    Xr,
    yr,
    test_size=TEST_SIZE,
    random_state=SEMILLA,
)

scaler_regresor = StandardScaler()
xr_train_esc = scaler_regresor.fit_transform(xr_train)
xr_test_esc = scaler_regresor.transform(xr_test)

modelo_rfr = RandomForestRegressor(
    random_state=SEMILLA
)
modelo_rfr.fit(xr_train_esc, yr_train)

modelo_mlp = MLPRegressor(
    random_state=SEMILLA,
    max_iter=500,
)
modelo_mlp.fit(xr_train_esc, yr_train)

r2_rfr_train = modelo_rfr.score(xr_train_esc, yr_train)
r2_rfr_test = modelo_rfr.score(xr_test_esc, yr_test)

r2_mlp_train = modelo_mlp.score(xr_train_esc, yr_train)
r2_mlp_test = modelo_mlp.score(xr_test_esc, yr_test)

print("Random Forest Regressor")
print(f"R² Train: {r2_rfr_train:.4f}")
print(f"R² Test:  {r2_rfr_test:.4f}")

print("\nMLP Regressor")
print(f"R² Train: {r2_mlp_train:.4f}")
print(f"R² Test:  {r2_mlp_test:.4f}")

resultados = {
    "Random Forest Regressor": (
        modelo_rfr,
        r2_rfr_test,
    ),
    "MLP Regressor": (
        modelo_mlp,
        r2_mlp_test,
    ),
}

nombre_ganador = max(
    resultados,
    key=lambda nombre: resultados[nombre][1],
)
modelo_ganador, r2_ganador = resultados[nombre_ganador]

joblib.dump(
    modelo_ganador,
    "models/regresor_presion.pkl",
)
joblib.dump(
    scaler_regresor,
    "models/scaler_regresor.pkl",
)

print(
    f"\nModelo ganador: {nombre_ganador} "
    f"(R² Test={r2_ganador:.4f})"
)
print("Archivos exportados:")
print("- data/cardio_clean.csv")
print("- models/regresor_presion.pkl")
print("- models/scaler_regresor.pkl")
