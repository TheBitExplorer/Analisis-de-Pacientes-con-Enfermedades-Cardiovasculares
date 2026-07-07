from dash import dcc, html

TABS = [
    ("/", "📊 Dashboard Cardiovascular"),
    ("/mapa", "🗺️ Mapa de Ingresos"),
]


def build_nav(active_path):
    """Barra de pestañas compartida entre las páginas del dashboard.

    Usa dcc.Link para navegar sin recargar la página y sin abrir
    ventanas o pestañas nuevas del navegador (todo queda dentro del
    mismo servicio de Render).
    """
    return html.Div(
        [
            dcc.Link(
                label,
                href=path,
                className="nav-tab nav-tab-active" if path == active_path else "nav-tab",
            )
            for path, label in TABS
        ],
        className="nav-tabs-container",
    )
