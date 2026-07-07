from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html, register_page

from nav_bar import build_nav

register_page(__name__, path="/", title="Dashboard Cardiovascular")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

df = pd.read_csv(DATA_DIR / "cardio_clean.csv")
modelo_regresion = joblib.load(MODELS_DIR / "regresor_presion.pkl")
scaler_regresion = joblib.load(MODELS_DIR / "scaler_regresor.pkl")

df["Diagnóstico"] = df["cardio"].map({0: "Sano", 1: "Enfermo"})

VARIABLES_BOXPLOT = {
    "age": "Edad (Años)",
    "ap_hi(mmHg)": "Presión Arterial Sistólica (mmHg)",
    "weight(kg)": "Peso (kg)",
}

VARIABLES_CATEGORICAS = {
    "cholesterol": "Nivel de Colesterol",
    "gluc": "Nivel de Glucosa",
    "smoke": "Hábito de Fumar",
    "active": "Actividad Física",
}

FEATURES_REG = ["age", "weight(kg)", "ap_lo(mmHg)", "cholesterol", "cardio"]

CORRELACION_AP_HI = {
    "ap_lo(mmHg)": 0.735960,
    "cardio": 0.428057,
    "weight(kg)": 0.270712,
    "age": 0.209196,
    "cholesterol": 0.195198,
}

conteo_cardio = df.groupby(["cardio", "Diagnóstico"]).size().reset_index(name="Pacientes")
fig_cardio = px.bar(
    conteo_cardio, x="cardio", y="Pacientes", color="Diagnóstico",
    title="Distribución de la Variable Objetivo (Cardio)",
    labels={"cardio": "Cardio (0: Sano, 1: Enfermo)"},
)
fig_cardio.update_layout(showlegend=False)

corr = df.corr(numeric_only=True)
fig_correlacion = px.imshow(
    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0, aspect="auto", title="Matriz de correlación",
)
fig_correlacion.update_layout(height=700)

df_scatter = df.sample(n=min(5000, len(df)), random_state=42)
fig_presiones = px.scatter(
    df_scatter, x="ap_hi(mmHg)", y="ap_lo(mmHg)", color="Diagnóstico",
    symbol="Diagnóstico", opacity=0.7,
    hover_data=["age", "weight(kg)", "cholesterol"],
    labels={
        "ap_hi(mmHg)": "Presión sistólica (mmHg)",
        "ap_lo(mmHg)": "Presión diastólica (mmHg)",
    },
    title="Relación entre la presión sistólica y diastólica según el diagnóstico",
    render_mode="webgl",
)

corr_reg = pd.DataFrame({
    "Variable": list(CORRELACION_AP_HI.keys()),
    "Correlación": list(CORRELACION_AP_HI.values()),
}).sort_values("Correlación")
fig_corr_reg = px.bar(
    corr_reg, x="Correlación", y="Variable", orientation="h", text="Correlación",
    title="Variables con mayor correlación respecto a ap_hi(mmHg)",
)
fig_corr_reg.update_traces(texttemplate="%{text:.3f}", textposition="outside")


def clasificar_presion(ap_hi):
    if ap_hi < 120: return "Normal"
    if ap_hi <= 129: return "Elevada"
    if ap_hi <= 139: return "Hipertensión Grado 1"
    if ap_hi <= 180: return "Hipertensión Grado 2"
    return "Crisis Hipertensiva"


def tarjeta_indicador(titulo, valor):
    return html.Div([
        html.P(titulo, className="kpi-title"),
        html.H3(valor, className="kpi-value"),
    ], className="kpi-card")


def campo_numero(etiqueta, component_id, value, minimo=None, maximo=None, step=1):
    return html.Div([
        html.Label(etiqueta),
        dcc.Input(id=component_id, type="number", value=value, min=minimo,
                  max=maximo, step=step, className="input-control"),
    ], className="form-field")


def campo_dropdown(etiqueta, component_id, options, value):
    return html.Div([
        html.Label(etiqueta),
        dcc.Dropdown(id=component_id, options=options, value=value, clearable=False),
    ], className="form-field")


layout = html.Div([
    build_nav("/"),

    html.Header([
        html.H1("Dashboard de Análisis Cardiovascular"),
        html.P(
            "Visualizaciones obtenidas del notebook del proyecto y uso del "
            "modelo de regresión con las variables de mayor correlación."
        ),
    ], className="page-header"),

    html.Main([
        html.Section([
            tarjeta_indicador("Registros analizados", f"{len(df):,}"),
            tarjeta_indicador("Edad promedio", f"{df['age'].mean():.1f} años"),
            tarjeta_indicador("Presión sistólica promedio", f"{df['ap_hi(mmHg)'].mean():.1f} mmHg"),
            tarjeta_indicador("Diagnóstico cardiovascular positivo", f"{df['cardio'].mean() * 100:.1f}%"),
        ], className="kpi-grid"),

        html.Section([
            html.H2("1. Visualizaciones del notebook"),
            html.Div([
                html.Div([dcc.Graph(figure=fig_cardio)], className="panel"),
                html.Div([
                    html.Label("Seleccione una variable clínica del notebook"),
                    dcc.Dropdown(
                        id="variable-boxplot",
                        options=[{"label": v, "value": k} for k, v in VARIABLES_BOXPLOT.items()],
                        value="age", clearable=False,
                    ),
                    dcc.Graph(id="grafica-boxplot"),
                ], className="panel"),
            ], className="two-column-grid"),

            html.Div([
                html.Div([
                    html.Label("Seleccione una variable categórica del notebook"),
                    dcc.Dropdown(
                        id="variable-categorica",
                        options=[{"label": v, "value": k} for k, v in VARIABLES_CATEGORICAS.items()],
                        value="cholesterol", clearable=False,
                    ),
                    dcc.Graph(id="grafica-categorica"),
                ], className="panel"),
                html.Div([dcc.Graph(figure=fig_presiones)], className="panel"),
            ], className="two-column-grid"),

            html.Div([dcc.Graph(figure=fig_correlacion)], className="panel"),
        ]),

        html.Section([
            html.H2("2. Variables de mayor correlación para la regresión"),
            html.Div([dcc.Graph(figure=fig_corr_reg)], className="panel"),
        ]),

        html.Section([
            html.H2("3. Predicción de presión sistólica"),
            html.P("El control usa las cinco variables de mayor correlación con ap_hi(mmHg), excluyendo la variable objetivo."),
            html.Div([
                campo_numero("Edad (age)", "reg-age", 50, 1, 120),
                campo_numero("Peso en kg (weight)", "reg-weight", 70, 40, 200, 0.1),
                campo_numero("Presión diastólica en mmHg (ap_lo)", "reg-ap-lo", 80, 40, 149),
                campo_dropdown("Colesterol", "reg-cholesterol", [
                    {"label": "1 - Normal", "value": 1},
                    {"label": "2 - Alto", "value": 2},
                    {"label": "3 - Muy alto", "value": 3},
                ], 1),
                campo_dropdown("Diagnóstico cardiovascular", "reg-cardio", [
                    {"label": "0 - Sano", "value": 0},
                    {"label": "1 - Enfermo", "value": 1},
                ], 0),
            ], className="prediction-form"),
            html.Button("Predecir presión sistólica", id="btn-predecir", n_clicks=0, className="primary-button"),
            html.Div(id="resultado-prediccion", className="prediction-result"),
            html.P(
                "Modelo académico de análisis de datos. El resultado no constituye un diagnóstico médico.",
                className="small-note",
            ),
        ], className="panel prediction-section"),
    ], className="page-content"),
])


@callback(Output("grafica-boxplot", "figure"), Input("variable-boxplot", "value"))
def actualizar_boxplot(variable):
    etiqueta = VARIABLES_BOXPLOT[variable]
    fig = px.box(df, x="cardio", y=variable, color="Diagnóstico",
                 title=f"Distribución de {etiqueta} por Diagnóstico",
                 labels={"cardio": "Cardio (0: Sano, 1: Enfermo)", variable: etiqueta})
    fig.update_layout(showlegend=False)
    return fig


@callback(Output("grafica-categorica", "figure"), Input("variable-categorica", "value"))
def actualizar_categorica(variable):
    resumen = df.groupby([variable, "Diagnóstico"]).size().reset_index(name="Pacientes")
    fig = px.bar(resumen, x=variable, y="Pacientes", color="Diagnóstico",
                 barmode="group", title=f"{VARIABLES_CATEGORICAS[variable]} vs Diagnóstico")
    etiquetas_x = {
        "cholesterol": "Colesterol (1: Normal, 2: Alto, 3: Muy Alto)",
        "gluc": "Glucosa (1: Normal, 2: Alto, 3: Muy Alto)",
        "smoke": "Fumador (0: No, 1: Sí)",
        "active": "Activo Físicamente (0: No, 1: Sí)",
    }
    fig.update_layout(xaxis_title=etiquetas_x[variable], yaxis_title="Pacientes")
    return fig


@callback(
    Output("resultado-prediccion", "children"),
    Input("btn-predecir", "n_clicks"),
    State("reg-age", "value"), State("reg-weight", "value"),
    State("reg-ap-lo", "value"), State("reg-cholesterol", "value"),
    State("reg-cardio", "value"),
    prevent_initial_call=True,
)
def predecir_presion(n_clicks, age, weight, ap_lo, cholesterol, cardio):
    valores = [age, weight, ap_lo, cholesterol, cardio]
    if any(v is None for v in valores):
        return html.P("Complete todos los campos antes de predecir.")
    instancia = pd.DataFrame([valores], columns=FEATURES_REG)
    try:
        prediccion = float(modelo_regresion.predict(scaler_regresion.transform(instancia))[0])
    except Exception as e:
        return html.Div([html.Strong("No se pudo ejecutar la predicción."), html.P(str(e))])
    return html.Div([
        html.H3(f"{prediccion:.2f} mmHg"),
        html.P(f"Categoría estimada: {clasificar_presion(prediccion)}"),
    ])
