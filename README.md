# Dashboard Cardiovascular

Repositorio preparado para desplegar el dashboard del proyecto final en Render.

## Estructura

```text
.
├── app.py
├── exportar_dashboard.py
├── requirements.txt
├── render.yaml
├── assets/
│   └── style.css
├── data/
│   ├── cardio_clean.csv
│   ├── datos_inec_ingresos_distritos_limpio.csv
│   └── geoBoundaries-PAN-ADM2-all.zip
└── models/
    ├── regresor_presion.pkl
    └── scaler_regresor.pkl
```

## Antes de desplegar

Ejecuta `exportar_dashboard.py` en el entorno donde tienes el dataset cardiovascular.
Luego copia los archivos generados a `data/` y `models/`.

También copia a `data/` los dos archivos usados por el mapa en el notebook:

- `datos_inec_ingresos_distritos_limpio.csv`
- `geoBoundaries-PAN-ADM2-all.zip`

## GitHub

No subas la carpeta contenedora adicional si GitHub te permite cargar archivos.
La raíz del repositorio debe mostrar directamente:

- `app.py`
- `requirements.txt`
- `render.yaml`

## Render

El archivo `render.yaml` ya define:

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:server`

En Render conecta este repositorio de GitHub y crea el servicio usando el Blueprint
o un Web Service con los mismos comandos.
