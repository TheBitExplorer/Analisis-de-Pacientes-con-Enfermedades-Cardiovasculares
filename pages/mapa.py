import unicodedata
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, State, callback, callback_context, ALL, register_page

from nav_bar import build_nav

register_page(__name__, path="/mapa", title="Mapa de Ingresos — Panamá")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RUTA_MAPA = DATA_DIR / "geoBoundaries-PAN-ADM2-all.zip"
RUTA_INEC = DATA_DIR / "datos_inec_ingresos_distritos_limpio.csv"


def normalize_text(text):
    if isinstance(text, str):
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
            .upper()
            .strip()
        )
    return text


CORRECCIONES = {
    "SAN FALIX": "SAN FELIX", "BARAO": "BARU", "BOQUERA3N": "BOQUERON",
    "TOLA": "TOLE", "ANTA3N": "ANTON", "PENONOMA": "PENONOME",
    "MIRONA3": "MIRONO", "KANKINTAO": "KANKINTU", "KUSAPAN": "KUSAPIN",
    "AA14RA14M": "MARIATO", "MA14NA": "MUNA", "RAO DE JESAOS": "RIO DE JESUS",
    "CHITRA": "CHITRE", "PESA": "PESE", "OCAO": "CEMACO", "SAMBAO": "SAMBU",
    "COLA3N": "COLON", "CAMACO": "CEMACO",
    "CHIRIQUA GRANDE": "CHIRIQUI GRANDE", "SANTA FA": "SANTA FE",
    "CAAAZAS": "CANAZAS", "SANTA MARAA": "SANTA MARIA",
    "PEDASA": "PEDASI", "POCRA": "POCRI", "TONOSA": "TONOSI", "GUARARA": "GUARARE",
}

gdf_raw = gpd.read_file(f"zip://{RUTA_MAPA.as_posix()}", layer="geoBoundaries-PAN-ADM2")
df_inec = pd.read_csv(RUTA_INEC, sep=",")

gdf_raw["distrito_mapa"] = gdf_raw["shapeName"].apply(normalize_text).replace(CORRECCIONES)
df_inec["distrito_inec"] = df_inec["Nombre Distrito"].apply(normalize_text)

gdf_merged = gdf_raw.merge(
    df_inec, left_on="distrito_mapa", right_on="distrito_inec", how="left"
).reset_index(drop=True)
gdf_merged["idx"] = gdf_merged.index.astype(str)
gdf_merged["geometry"] = gdf_merged["geometry"].simplify(tolerance=0.01, preserve_topology=True)

GEOJSON_COMPLETO = json.loads(gdf_merged.to_json())

CENTROS_PROVINCIA = {}
for prov, group in gdf_merged.dropna(subset=["Nombre Provincia"]).groupby("Nombre Provincia"):
    union = group.geometry.unary_union
    centroid = union.centroid
    minx, miny, maxx, maxy = union.bounds
    CENTROS_PROVINCIA[prov] = {
        "lat": round(centroid.y, 3),
        "lon": round(centroid.x, 3),
        "bounds": (minx, miny, maxx, maxy),
    }

PROVINCIAS = ["Todas las provincias"] + sorted(df_inec["Nombre Provincia"].dropna().unique().tolist())

CARD = {"backgroundColor": "white", "borderRadius": "8px", "padding": "15px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"}
BTN_BASE = {
    "border": "1.5px solid #1a3a5c", "borderRadius": "20px",
    "padding": "5px 14px", "cursor": "pointer",
    "fontSize": "12px", "fontWeight": "bold",
}
BTN_ACTIVO   = {**BTN_BASE, "backgroundColor": "#1a3a5c", "color": "white"}
BTN_INACTIVO = {**BTN_BASE, "backgroundColor": "white",   "color": "#1a3a5c"}

layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f4f6f9",
           "padding": "12px 2%", "maxWidth": "100%", "margin": "0 auto"},
    children=[
        build_nav("/mapa"),

        html.Div(
            style={"backgroundColor": "#1a3a5c", "padding": "16px 30px",
                   "borderRadius": "8px", "marginBottom": "14px"},
            children=[
                html.H1("Panamá — Ingresos Promedio por Distrito",
                        style={"color": "white", "margin": 0, "fontSize": "22px"}),
                html.P("Fuente: INEC — Censos de Población y Vivienda 2023",
                       style={"color": "#a0b8d0", "margin": "4px 0 0 0", "fontSize": "12px"}),
            ],
        ),

        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "12px",
                   "marginBottom": "10px", "flexWrap": "wrap"},
            children=[
                html.Label("Paleta:", style={"fontWeight": "bold", "fontSize": "13px"}),
                dcc.Dropdown(
                    id="mapa-dropdown-paleta",
                    options=[
                        {"label": "Rojo-Verde", "value": "RdYlGn"},
                        {"label": "Viridis",    "value": "Viridis"},
                        {"label": "Blues",      "value": "Blues"},
                        {"label": "YlOrRd",     "value": "YlOrRd"},
                        {"label": "Plasma",     "value": "Plasma"},
                        {"label": "Turbo",      "value": "Turbo"},
                    ],
                    value="RdYlGn", clearable=False, style={"width": "180px"},
                ),
                html.Div(id="mapa-info-click",
                         style={"fontSize": "12px", "color": "#888", "fontStyle": "italic", "marginLeft": "auto"},
                         children="Haz clic en un distrito del mapa para ver su detalle."),
            ],
        ),

        dcc.Store(id="mapa-provincia-store", data="Todas las provincias"),

        html.Div(id="mapa-panel-resumen", style={"marginBottom": "10px"}),

        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "12px", "alignItems": "center"},
            children=[
                html.Div(
                    id="mapa-contenedor-botones",
                    style={"display": "grid", "gridTemplateColumns": "repeat(5, auto)",
                           "gap": "5px", "flexShrink": "0"},
                    children=[
                        html.Button(
                            p if p != "Todas las provincias" else "🗺️ Todas",
                            id={"type": "mapa-btn-provincia", "index": p},
                            n_clicks=0,
                            style=BTN_ACTIVO if p == "Todas las provincias" else BTN_INACTIVO,
                        )
                        for p in PROVINCIAS
                    ],
                ),
                html.Div(id="mapa-panel-distrito", style={"flex": "1", "minWidth": "0"}),
            ],
        ),

        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "14px", "alignItems": "stretch"},
            children=[
                html.Div(style={**CARD, "flex": "3", "minWidth": "0"},
                         children=[dcc.Graph(id="mapa-choropleth",
                                             style={"height": "52vh", "minHeight": "400px"},
                                             config={"scrollZoom": True})]),
                html.Div(style={**CARD, "flex": "2", "minWidth": "0"},
                         children=[dcc.Graph(id="mapa-grafica-barras",
                                             style={"height": "52vh", "minHeight": "400px"})]),
            ],
        ),
    ],
)


@callback(
    Output("mapa-provincia-store", "data"),
    Input({"type": "mapa-btn-provincia", "index": ALL}, "n_clicks"),
    Input("mapa-choropleth", "clickData"),
    State("mapa-provincia-store", "data"),
    prevent_initial_call=True,
)
def actualizar_store(_, click_mapa, provincia_actual):
    triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    if "mapa-btn-provincia" in triggered:
        return json.loads(triggered.split(".")[0])["index"]
    if "mapa-choropleth" in triggered and click_mapa:
        fila = gdf_merged.iloc[int(click_mapa["points"][0]["location"])]
        prov = fila.get("Nombre Provincia", None)
        if pd.notna(prov):
            return prov
    return provincia_actual


@callback(
    Output({"type": "mapa-btn-provincia", "index": ALL}, "style"),
    Input("mapa-provincia-store", "data"),
)
def resaltar_boton(provincia_activa):
    return [BTN_ACTIVO if p == provincia_activa else BTN_INACTIVO for p in PROVINCIAS]


@callback(
    Output("mapa-choropleth", "figure"),
    Input("mapa-dropdown-paleta", "value"),
    Input("mapa-provincia-store", "data"),
)
def actualizar_mapa(paleta, provincia):
    if provincia == "Todas las provincias":
        df_plot = gdf_merged
        geojson = GEOJSON_COMPLETO
        zoom, center = 6.8, {"lat": 8.4, "lon": -80.1}
    else:
        df_plot = gdf_merged[gdf_merged["Nombre Provincia"] == provincia]
        geojson = json.loads(df_plot.to_json())
        ZOOM_GENERAL = {"DARIÉN", "COLÓN", "COMARCA KUNA YALA", "COMARCA EMBERÁ"}
        if provincia in ZOOM_GENERAL:
            zoom, center = 6.8, {"lat": 8.4, "lon": -80.1}
        else:
            info = CENTROS_PROVINCIA.get(provincia, {"lat": 8.4, "lon": -80.1, "bounds": (-80.1, 8.4, -80.1, 8.4)})
            center = {"lat": info["lat"], "lon": info["lon"]}
            minx, miny, maxx, maxy = info["bounds"]
            span = max(maxx - minx, maxy - miny)
            zoom = 7.5 if span > 1.3 else 8.5

    fig = px.choropleth_mapbox(
        df_plot, geojson=geojson, locations=df_plot.index,
        color="Valor", hover_name="Nombre Distrito",
        hover_data={"Nombre Provincia": True, "Valor": ":,.0f"},
        color_continuous_scale=paleta, mapbox_style="carto-positron",
        zoom=zoom, center=center, opacity=0.75,
        labels={"Valor": "Ingreso (USD)"},
        title="Ingreso Promedio Mensual por Distrito (USD)",
        custom_data=["Nombre Distrito", "Nombre Provincia", "Valor"],
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>$%{customdata[2]:,.0f} / mes<extra></extra>"
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="USD"),
        clickmode="event+select",
        uirevision=provincia,
    )
    return fig


@callback(
    Output("mapa-grafica-barras", "figure"),
    Output("mapa-panel-resumen", "children"),
    Output("mapa-info-click", "children"),
    Output("mapa-panel-distrito", "children"),
    Input("mapa-dropdown-paleta", "value"),
    Input("mapa-provincia-store", "data"),
    Input("mapa-choropleth", "clickData"),
)
def actualizar_barras(paleta, provincia, click_data):
    info_msg = "Haz clic en un distrito del mapa para ver su detalle."
    panel_dist = html.Div()
    triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    fila_sel = None
    if "mapa-choropleth" in triggered and click_data:
        fila_sel = gdf_merged.iloc[int(click_data["points"][0]["location"])]
        if str(fila_sel.get("Nombre Provincia", None)) != provincia:
            fila_sel = None

    if fila_sel is not None:
        distrito_nombre  = fila_sel.get("Nombre Distrito", "—")
        provincia_nombre = fila_sel.get("Nombre Provincia", "—")
        valor            = fila_sel.get("Valor", None)
        fuente           = fila_sel.get("Fuente", "—")
        valor_str        = f"${valor:,.0f} / mes" if pd.notna(valor) else "Sin datos"
        info_msg         = f"Distrito seleccionado: {distrito_nombre} ({provincia_nombre})"
        panel_dist = html.Div(
            style={**CARD, "borderLeft": "4px solid #1a3a5c", "padding": "6px 10px"},
            children=[
                html.P(f"📍 {distrito_nombre} • {provincia_nombre} • {valor_str}",
                       style={"margin": 0, "fontSize": "14px", "fontWeight": "bold", "color": "#1a3a5c"}),
                html.P(str(fuente), style={"margin": "1px 0 0 0", "fontSize": "12px", "color": "#888"}),
            ],
        )

    df_bar = (
        df_inec if provincia == "Todas las provincias"
        else df_inec[df_inec["Nombre Provincia"] == provincia]
    )
    df_bar = df_bar.dropna(subset=["Valor"]).sort_values("Valor", ascending=True)

    if provincia == "Todas las provincias":
        val_min = df_bar["Valor"].min()
        val_max = df_bar["Valor"].max()
        orden_provincias = (
            df_bar.groupby("Nombre Provincia")["Valor"]
            .median().sort_values(ascending=True).index.tolist()
        )
        fig_barras = px.scatter(
            df_bar, x="Valor", y="Nombre Provincia",
            hover_name="Nombre Distrito",
            hover_data={"Valor": ":,.0f", "Nombre Provincia": False},
            color="Valor", color_continuous_scale=paleta,
            range_color=[val_min, val_max],
            labels={"Valor": "Ingreso Promedio (USD)", "Nombre Provincia": "Provincia"},
            title="Ingresos por Distrito — agrupado por Provincia",
        )
        fig_barras.update_traces(marker={"size": 9, "opacity": 0.85})
        fig_barras.update_layout(
            margin={"r": 20, "t": 40, "l": 10, "b": 20},
            showlegend=False, coloraxis_showscale=False,
            xaxis={"tickfont": {"size": 9}},
            yaxis={"categoryorder": "array", "categoryarray": orden_provincias, "tickfont": {"size": 9}},
            font={"size": 10}, uirevision=provincia,
        )
    else:
        rango_min = max(0, df_bar["Valor"].min() * 0.85) if len(df_bar) > 0 else 0
        rango_max = df_bar["Valor"].max() * 1.08 if len(df_bar) > 0 else 1000
        fig_barras = px.bar(
            df_bar, x="Valor", y="Nombre Distrito", orientation="h",
            color="Valor", color_continuous_scale=paleta,
            labels={"Valor": "Ingreso Promedio (USD)", "Nombre Distrito": "Distrito"},
            title=f"Ingreso Promedio Mensual — {provincia}",
            text="Valor",
        )
        fig_barras.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                                 textfont_size=9, width=0.5)
        fig_barras.update_layout(
            margin={"r": 60, "t": 40, "l": 10, "b": 20},
            showlegend=False, coloraxis_showscale=False,
            xaxis={"range": [rango_min, rango_max], "tickfont": {"size": 9}},
            yaxis={"categoryorder": "total ascending", "tickfont": {"size": 9}},
            font={"size": 10}, autosize=True,
        )

    if len(df_bar) > 0:
        mayor    = df_bar.loc[df_bar["Valor"].idxmax()]
        menor    = df_bar.loc[df_bar["Valor"].idxmin()]
        promedio = df_bar["Valor"].mean()

        def tarjeta(label, nombre, val_str, color):
            return html.Div(style={**CARD, "flex": "1", "minWidth": "180px", "borderTop": f"4px solid {color}"}, children=[
                html.P(label, style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px 0",
                                     "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                html.P(nombre, style={"fontSize": "13px", "fontWeight": "bold", "color": "#1a3a5c", "margin": "0 0 2px 0"}),
                html.P(val_str, style={"fontSize": "17px", "fontWeight": "bold", "color": color, "margin": 0}),
            ])

        resumen = html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
            tarjeta("📊 Distritos analizados", f"{len(df_bar)} distritos", f"Promedio: ${promedio:,.0f}", "#1a3a5c"),
            tarjeta("🏆 Mayor ingreso", mayor["Nombre Distrito"], f"${mayor['Valor']:,.0f} / mes", "#27ae60"),
            tarjeta("📉 Menor ingreso", menor["Nombre Distrito"], f"${menor['Valor']:,.0f} / mes", "#e74c3c"),
        ])
    else:
        resumen = html.P("Sin datos disponibles.")

    return fig_barras, resumen, info_msg, panel_dist
