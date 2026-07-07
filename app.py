from pathlib import Path
import json
import unicodedata

import geopandas as gpd
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CARDIO_CSV = DATA_DIR / "cardio_clean.csv"
MAP_ZIP = DATA_DIR / "geoBoundaries-PAN-ADM2-all.zip"
INEC_CSV = DATA_DIR / "datos_inec_ingresos_distritos_limpio.csv"
REG_MODEL = MODELS_DIR / "regresor_presion.pkl"
REG_SCALER = MODELS_DIR / "scaler_regresor.pkl"

ARCHIVOS_REQUERIDOS = [
    CARDIO_CSV,
    MAP_ZIP,
    INEC_CSV,
    REG_MODEL,
    REG_SCALER,
]

faltantes = [
    str(ruta.relative_to(BASE_DIR))
    for ruta in ARCHIVOS_REQUERIDOS
    if not ruta.exists()
]

if faltantes:
    raise FileNotFoundError(
        "Faltan archivos requeridos:\n- " + "\n- ".join(faltantes)
    )


# ============================================================
# CARGA DE DATOS Y MODELO
# ============================================================

df = pd.read_csv(CARDIO_CSV)
modelo_regresion = joblib.load(REG_MODEL)
scaler_regresion = joblib.load(REG_SCALER)

df["Diagnóstico"] = df["cardio"].map({0: "Sano", 1: "Enfermo"})


# ============================================================
# VARIABLES DEL NOTEBOOK
# ============================================================

# Sección 2.3 del notebook: boxplots clínicos
VARIABLES_BOXPLOT = {
    "age": "Edad (Años)",
    "ap_hi(mmHg)": "Presión Arterial Sistólica (mmHg)",
    "weight(kg)": "Peso (kg)",
}

# Sección 2.4 del notebook: categorías agrupadas
VARIABLES_CATEGORICAS = {
    "cholesterol": "Nivel de Colesterol",
    "gluc": "Nivel de Glucosa",
    "smoke": "Hábito de Fumar",
    "active": "Actividad Física",
}

# Sección 4 del notebook:
# las 5 variables usadas para predecir ap_hi.
# Son las variables con mayor correlación con ap_hi, excluyendo la propia variable objetivo.
FEATURES_REG = [
    "age",
    "weight(kg)",
    "ap_lo(mmHg)",
    "cholesterol",
    "cardio",
]

CORRELACION_AP_HI = {
    "ap_lo(mmHg)": 0.735960,
    "cardio": 0.428057,
    "weight(kg)": 0.270712,
    "age": 0.209196,
    "cholesterol": 0.195198,
}


# ============================================================
# GRÁFICAS DEL NOTEBOOK
# ============================================================

# Gráfica equivalente al countplot de la variable cardio
conteo_cardio = (
    df.groupby(["cardio", "Diagnóstico"])
    .size()
    .reset_index(name="Pacientes")
)

fig_cardio = px.bar(
    conteo_cardio,
    x="cardio",
    y="Pacientes",
    color="Diagnóstico",
    title="Distribución de la Variable Objetivo (Cardio)",
    labels={"cardio": "Cardio (0: Sano, 1: Enfermo)"},
)
fig_cardio.update_layout(showlegend=False)


# Sección 2: matriz de correlación
corr = df.corr(numeric_only=True)

fig_correlacion = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    aspect="auto",
    title="Matriz de correlación",
)
fig_correlacion.update_layout(height=700)


# Sección 2: scatter presión sistólica vs diastólica

# Muestra representativa para la visualización del dashboard
df_scatter = df.sample(
    n=min(5000, len(df)),
    random_state=42
)

fig_presiones = px.scatter(
    df_scatter,
    x="ap_hi(mmHg)",
    y="ap_lo(mmHg)",
    color="Diagnóstico",
    symbol="Diagnóstico",
    opacity=0.7,
    hover_data=[
        "age",
        "weight(kg)",
        "cholesterol"
    ],
    labels={
        "ap_hi(mmHg)": "Presión sistólica (mmHg)",
        "ap_lo(mmHg)": "Presión diastólica (mmHg)",
        "Diagnóstico": "Diagnóstico",
    },
    title=(
        "Relación entre la presión sistólica y "
        "diastólica según el diagnóstico"
    ),
    render_mode="webgl",
)


# Correlaciones con ap_hi para justificar las entradas del modelo
corr_reg = pd.DataFrame(
    {
        "Variable": list(CORRELACION_AP_HI.keys()),
        "Correlación": list(CORRELACION_AP_HI.values()),
    }
).sort_values("Correlación")

fig_corr_reg = px.bar(
    corr_reg,
    x="Correlación",
    y="Variable",
    orientation="h",
    text="Correlación",
    title="Variables con mayor correlación respecto a ap_hi(mmHg)",
)
fig_corr_reg.update_traces(texttemplate="%{text:.3f}", textposition="outside")


# ============================================================
# MAPA DEL NOTEBOOK
# ============================================================

def normalizar_texto(texto):
    if isinstance(texto, str):
        return (
            unicodedata.normalize("NFKD", texto)
            .encode("ascii", "ignore")
            .decode("utf-8")
            .upper()
            .strip()
        )
    return texto


CORRECCIONES_DISTRITOS = {
    "SAN FALIX": "SAN FELIX",
    "BARAO": "BARU",
    "BOQUERA3N": "BOQUERON",
    "TOLA": "TOLE",
    "ANTA3N": "ANTON",
    "PENONOMA": "PENONOME",
    "MIRONA3": "MIRONO",
    "KANKINTAO": "KANKINTU",
    "KUSAPAN": "KUSAPIN",
    "AA14RA14M": "MARIATO",
    "MA14NA": "MUNA",
    "RAO DE JESAOS": "RIO DE JESUS",
    "CHITRA": "CHITRE",
    "PESA": "PESE",
    "OCAO": "CEMACO",
    "SAMBAO": "SAMBU",
    "COLA3N": "COLON",
    "CAMACO": "CEMACO",
    "CHIRIQUA GRANDE": "CHIRIQUI GRANDE",
    "SANTA FA": "SANTA FE",
    "CAAAZAS": "CANAZAS",
    "SANTA MARAA": "SANTA MARIA",
    "PEDASA": "PEDASI",
    "POCRA": "POCRI",
    "TONOSA": "TONOSI",
    "GUARARA": "GUARARE",
}

gdf_mapa = gpd.read_file(f"zip://{MAP_ZIP.as_posix()}")
df_inec = pd.read_csv(INEC_CSV)

gdf_mapa["distrito_mapa"] = (
    gdf_mapa["shapeName"]
    .apply(normalizar_texto)
    .replace(CORRECCIONES_DISTRITOS)
)
df_inec["distrito_inec"] = df_inec["Nombre Distrito"].apply(normalizar_texto)

df_inec["Valor"] = pd.to_numeric(
    df_inec["Valor"].astype(str).str.replace(",", "", regex=False),
    errors="coerce",
)

gdf_merged = gdf_mapa.merge(
    df_inec,
    left_on="distrito_mapa",
    right_on="distrito_inec",
    how="left",
).reset_index(drop=True)

gdf_merged["map_id"] = gdf_merged.index.astype(str)
geojson_panama = json.loads(gdf_merged.to_json())

fig_mapa = px.choropleth_map(
    gdf_merged,
    geojson=geojson_panama,
    locations="map_id",
    featureidkey="properties.map_id",
    color="Valor",
    hover_name="shapeName",
    hover_data={"Valor": ":.2f", "map_id": False},
    labels={"Valor": "Ingreso promedio mensual por persona (USD)"},
    center={"lat": 8.5, "lon": -80.0},
    zoom=5.8,
    map_style="carto-positron",
    title="Ingreso Promedio Mensual por Persona por Distrito en Panamá (INEC 2023)",
)
fig_mapa.update_layout(height=650, margin={"r": 0, "t": 50, "l": 0, "b": 0})

resumen_mapa = (
    gdf_merged[["shapeName", "Valor"]]
    .dropna()
    .sort_values("Valor", ascending=False)
    .head(10)
    .sort_values("Valor")
)

fig_resumen_mapa = px.bar(
    resumen_mapa,
    x="Valor",
    y="shapeName",
    orientation="h",
    title="10 distritos con mayor ingreso promedio mensual por persona",
    labels={
        "Valor": "Ingreso promedio mensual por persona (USD)",
        "shapeName": "Distrito",
    },
)


# ============================================================
# AUXILIARES
# ============================================================

def clasificar_presion(ap_hi):
    if ap_hi < 120:
        return "Normal"
    if ap_hi <= 129:
        return "Elevada"
    if ap_hi <= 139:
        return "Hipertensión Grado 1"
    if ap_hi <= 180:
        return "Hipertensión Grado 2"
    return "Crisis Hipertensiva"


def tarjeta_indicador(titulo, valor):
    return html.Div(
        [
            html.P(titulo, className="kpi-title"),
            html.H3(valor, className="kpi-value"),
        ],
        className="kpi-card",
    )


def campo_numero(etiqueta, component_id, value, minimo=None, maximo=None, step=1):
    return html.Div(
        [
            html.Label(etiqueta),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimo,
                max=maximo,
                step=step,
                className="input-control",
            ),
        ],
        className="form-field",
    )


def campo_dropdown(etiqueta, component_id, options, value):
    return html.Div(
        [
            html.Label(etiqueta),
            dcc.Dropdown(
                id=component_id,
                options=options,
                value=value,
                clearable=False,
            ),
        ],
        className="form-field",
    )


# ============================================================
# APP
# ============================================================

app = Dash(__name__)
server = app.server
app.title = "Dashboard Cardiovascular"

app.layout = html.Div(
    [
        html.Header(
            [
                html.H1("Dashboard de Análisis Cardiovascular"),
                html.P(
                    "Visualizaciones obtenidas del notebook del proyecto y uso del "
                    "modelo de regresión con las variables de mayor correlación."
                ),
            ],
            className="page-header",
        ),

        html.Main(
            [
                html.Section(
                    [
                        tarjeta_indicador("Registros analizados", f"{len(df):,}"),
                        tarjeta_indicador("Edad promedio", f"{df['age'].mean():.1f} años"),
                        tarjeta_indicador(
                            "Presión sistólica promedio",
                            f"{df['ap_hi(mmHg)'].mean():.1f} mmHg",
                        ),
                        tarjeta_indicador(
                            "Diagnóstico cardiovascular positivo",
                            f"{df['cardio'].mean() * 100:.1f}%",
                        ),
                    ],
                    className="kpi-grid",
                ),

                html.Section(
                    [
                        html.H2("1. Visualizaciones del notebook"),

                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(figure=fig_cardio)],
                                    className="panel",
                                ),

                                html.Div(
                                    [
                                        html.Label(
                                            "Seleccione una variable clínica del notebook"
                                        ),
                                        dcc.Dropdown(
                                            id="variable-boxplot",
                                            options=[
                                                {"label": etiqueta, "value": columna}
                                                for columna, etiqueta in VARIABLES_BOXPLOT.items()
                                            ],
                                            value="age",
                                            clearable=False,
                                        ),
                                        dcc.Graph(id="grafica-boxplot"),
                                    ],
                                    className="panel",
                                ),
                            ],
                            className="two-column-grid",
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Seleccione una variable categórica del notebook"
                                        ),
                                        dcc.Dropdown(
                                            id="variable-categorica",
                                            options=[
                                                {"label": etiqueta, "value": columna}
                                                for columna, etiqueta in VARIABLES_CATEGORICAS.items()
                                            ],
                                            value="cholesterol",
                                            clearable=False,
                                        ),
                                        dcc.Graph(id="grafica-categorica"),
                                    ],
                                    className="panel",
                                ),

                                html.Div(
                                    [dcc.Graph(figure=fig_presiones)],
                                    className="panel",
                                ),
                            ],
                            className="two-column-grid",
                        ),

                        html.Div(
                            [dcc.Graph(figure=fig_correlacion)],
                            className="panel",
                        ),
                    ]
                ),

                html.Section(
                    [
                        html.H2("2. Variables de mayor correlación para la regresión"),
                        html.Div(
                            [dcc.Graph(figure=fig_corr_reg)],
                            className="panel",
                        ),
                    ]
                ),

                html.Section(
                    [
                        html.H2("3. Predicción de presión sistólica"),
                        html.P(
                            "El control usa las cinco variables de mayor correlación con "
                            "ap_hi(mmHg), excluyendo la variable objetivo."
                        ),

                        html.Div(
                            [
                                campo_numero(
                                    "Edad (age)",
                                    "reg-age",
                                    50,
                                    1,
                                    120,
                                ),
                                campo_numero(
                                    "Peso en kg (weight)",
                                    "reg-weight",
                                    70,
                                    40,
                                    200,
                                    0.1,
                                ),
                                campo_numero(
                                    "Presión diastólica en mmHg (ap_lo)",
                                    "reg-ap-lo",
                                    80,
                                    40,
                                    149,
                                ),
                                campo_dropdown(
                                    "Colesterol",
                                    "reg-cholesterol",
                                    [
                                        {"label": "1 - Normal", "value": 1},
                                        {"label": "2 - Alto", "value": 2},
                                        {"label": "3 - Muy alto", "value": 3},
                                    ],
                                    1,
                                ),
                                campo_dropdown(
                                    "Diagnóstico cardiovascular",
                                    "reg-cardio",
                                    [
                                        {"label": "0 - Sano", "value": 0},
                                        {"label": "1 - Enfermo", "value": 1},
                                    ],
                                    0,
                                ),
                            ],
                            className="prediction-form",
                        ),

                        html.Button(
                            "Predecir presión sistólica",
                            id="btn-predecir",
                            n_clicks=0,
                            className="primary-button",
                        ),

                        html.Div(
                            id="resultado-prediccion",
                            className="prediction-result",
                        ),

                        html.P(
                            "Modelo académico de análisis de datos. El resultado no "
                            "constituye un diagnóstico médico.",
                            className="small-note",
                        ),
                    ],
                    className="panel prediction-section",
                ),

                html.Section(
                    [
                        html.H2("4. Análisis sociodemográfico de Panamá"),
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(figure=fig_mapa)],
                                    className="panel",
                                ),
                                html.Div(
                                    [dcc.Graph(figure=fig_resumen_mapa)],
                                    className="panel",
                                ),
                            ],
                            className="map-grid",
                        ),
                    ]
                ),
            ],
            className="page-content",
        ),
    ]
)


# ============================================================
# CALLBACK 1
# BOXplots del notebook: age, ap_hi y weight por cardio
# ============================================================

@callback(
    Output("grafica-boxplot", "figure"),
    Input("variable-boxplot", "value"),
)
def actualizar_boxplot(variable):
    etiqueta = VARIABLES_BOXPLOT[variable]

    figura = px.box(
        df,
        x="cardio",
        y=variable,
        color="Diagnóstico",
        title=f"Distribución de {etiqueta} por Diagnóstico",
        labels={
            "cardio": "Cardio (0: Sano, 1: Enfermo)",
            variable: etiqueta,
        },
    )
    figura.update_layout(showlegend=False)
    return figura


# ============================================================
# CALLBACK 2
# Countplots categóricos del notebook
# ============================================================

@callback(
    Output("grafica-categorica", "figure"),
    Input("variable-categorica", "value"),
)
def actualizar_categorica(variable):
    resumen = (
        df.groupby([variable, "Diagnóstico"])
        .size()
        .reset_index(name="Pacientes")
    )

    figura = px.bar(
        resumen,
        x=variable,
        y="Pacientes",
        color="Diagnóstico",
        barmode="group",
        title=f"{VARIABLES_CATEGORICAS[variable]} vs Diagnóstico",
    )

    etiquetas_x = {
        "cholesterol": "Colesterol (1: Normal, 2: Alto, 3: Muy Alto)",
        "gluc": "Glucosa (1: Normal, 2: Alto, 3: Muy Alto)",
        "smoke": "Fumador (0: No, 1: Sí)",
        "active": "Activo Físicamente (0: No, 1: Sí)",
    }

    figura.update_layout(
        xaxis_title=etiquetas_x[variable],
        yaxis_title="Pacientes",
    )
    return figura


# ============================================================
# CALLBACK 3
# Modelo de regresión con las 5 variables más correlacionadas
# ============================================================

@callback(
    Output("resultado-prediccion", "children"),
    Input("btn-predecir", "n_clicks"),
    State("reg-age", "value"),
    State("reg-weight", "value"),
    State("reg-ap-lo", "value"),
    State("reg-cholesterol", "value"),
    State("reg-cardio", "value"),
    prevent_initial_call=True,
)
def predecir_presion(
    n_clicks,
    age,
    weight,
    ap_lo,
    cholesterol,
    cardio,
):
    valores = [age, weight, ap_lo, cholesterol, cardio]

    if any(valor is None for valor in valores):
        return html.P("Complete todos los campos antes de predecir.")

    # Mismo orden usado en la Sección 4 del notebook.
    instancia = pd.DataFrame(
        [valores],
        columns=FEATURES_REG,
    )

    try:
        instancia_escalada = scaler_regresion.transform(instancia)
        prediccion = float(
            modelo_regresion.predict(instancia_escalada)[0]
        )
    except Exception as error:
        return html.Div(
            [
                html.Strong("No se pudo ejecutar la predicción."),
                html.P(str(error)),
            ]
        )

    categoria = clasificar_presion(prediccion)

    return html.Div(
        [
            html.H3(f"{prediccion:.2f} mmHg"),
            html.P(f"Categoría estimada: {categoria}"),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
