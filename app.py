import dash
from dash import Dash, html

# ============================================================
# APP (router de Dash Pages)
# Las páginas reales viven en la carpeta pages/:
#   - pages/home.py -> "/"      Dashboard cardiovascular
#   - pages/mapa.py -> "/mapa"  Mapa de ingresos de Panamá
# Dash las descubre automáticamente gracias a use_pages=True.
# ============================================================

app = Dash(__name__, use_pages=True, pages_folder="pages")
server = app.server
app.title = "Dashboard Cardiovascular"

app.layout = html.Div(
    [
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
