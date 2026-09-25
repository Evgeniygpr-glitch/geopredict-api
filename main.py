```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import math
import os
import requests
import folium
import srtm


# =========================================================
# CONFIG
# =========================================================

APP_NAME = "GeoPredict"
VERSION = "0.1.0"

# Ромни
CENTER_LAT = 50.7460
CENTER_LON = 33.4747

# Радіус аналізу
DEFAULT_RADIUS_KM = 30.0

# Sentinel-2
COPERNICUS_CLIENT_ID = os.getenv(
    "COPERNICUS_CLIENT_ID"
)

COPERNICUS_CLIENT_SECRET = os.getenv(
    "COPERNICUS_CLIENT_SECRET"
)

# Максимальна хмарність сцени
MAX_CLOUD_COVER = 10


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title=APP_NAME,
    version=VERSION
)


# =========================================================
# SRTM
# =========================================================

elevation_data = srtm.get_data()


# =========================================================
# HELPERS
# =========================================================

def get_bbox(
    lat: float,
    lon: float,
    radius_km: float
):
    """
    Повертає bounding box навколо точки.
    """

    lat_delta = radius_km / 111.0

    lon_delta = radius_km / (
        111.0 *
        math.cos(
            math.radians(lat)
        )
    )

    return {
        "min_lat": lat - lat_delta,
        "max_lat": lat + lat_delta,
        "min_lon": lon - lon_delta,
        "max_lon": lon + lon_delta
    }


# =========================================================
# TERRAIN
# =========================================================

def load_dem(
    lat: float,
    lon: float,
    radius_km: float,
    steps: int = 180
):
    """
    Завантажує сітку висот SRTM.

    На першому етапі використовуємо SRTM.
    Пізніше можна замінити джерело DEM.
    """

    bbox = get_bbox(
        lat,
        lon,
        radius_km
    )

    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]

    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]

    lat_step = (
        max_lat - min_lat
    ) / (steps - 1)

    lon_step = (
        max_lon - min_lon
    ) / (steps - 1)

    grid = []
    lats = []
    lons = []

    for r in range(steps):

        current_lat = (
            min_lat +
            r * lat_step
        )

        lats.append(
            current_lat
        )

        row = []

        for c in range(steps):

            current_lon = (
                min_lon +
                c * lon_step
            )

            if r == 0:
                lons.append(
                    current_lon
                )

            elevation = (
                elevation_data.get_elevation(
                    current_lat,
                    current_lon
                )
            )

            row.append(
                elevation
            )

        grid.append(row)

    return (
        grid,
        lats,
        lons
    )


# =========================================================
# SLOPE
# =========================================================

def calculate_slope(
    grid,
    r,
    c,
    cell_size_m
):
    """
    Розрахунок крутизни схилу.
    """

    z1 = grid[r][c - 1]
    z2 = grid[r][c + 1]

    z3 = grid[r - 1][c]
    z4 = grid[r + 1][c]

    if None in (
        z1,
        z2,
        z3,
        z4
    ):
        return None

    dz_dx = (
        z2 - z1
    ) / (
        2 * cell_size_m
    )

    dz_dy = (
        z4 - z3
    ) / (
        2 * cell_size_m
    )

    slope = math.degrees(
        math.atan(
            math.sqrt(
                dz_dx ** 2 +
                dz_dy ** 2
            )
        )
    )

    return slope


# =========================================================
# ASPECT
# =========================================================

def calculate_aspect(
    grid,
    r,
    c,
    cell_size_m
):
    """
    Напрямок схилу.
    """

    z1 = grid[r][c - 1]
    z2 = grid[r][c + 1]

    z3 = grid[r - 1][c]
    z4 = grid[r + 1][c]

    if None in (
        z1,
        z2,
        z3,
        z4
    ):
        return None

    dz_dx = (
        z2 - z1
    ) / (
        2 * cell_size_m
    )

    dz_dy = (
        z4 - z3
    ) / (
        2 * cell_size_m
    )

    aspect = (
        math.degrees(
            math.atan2(
                -dz_dy,
                dz_dx
            )
        ) + 360
    ) % 360

    return aspect


# =========================================================
# CHERNYAKHIV PROFILE
# =========================================================

CHERNYAKHIV_WEIGHTS = {

    # Малі долини / початки ярів
    "valley_head": 0.25,

    # Близькість до малих водотоків
    "small_water": 0.20,

    # Придатність схилу
    "slope": 0.15,

    # Сонячна експозиція
    "sun": 0.10,

    # Захист від вітру
    "wind_shelter": 0.10,

    # Пасовища / відкрита територія
    "pasture": 0.08,

    # Давні/сучасні дороги
    "road": 0.05,

    # Відстань до великої річки
    "large_river": 0.07
}


# =========================================================
# CHERNYAKHIV SITE SCORE
# =========================================================

def calculate_chernyakhiv_score(
    slope,
    aspect
):
    """
    Поки що базова модель.

    ВАЖЛИВО:
    реальні valley_head, water, roads,
    pasture тощо підключимо окремими
    шарами.

    Тут навмисно НЕ вигадуємо
    археологічні дані.
    """

    if slope is None:
        return 0.0

    # -----------------------------------------------------
    # SLOPE
    # -----------------------------------------------------
    #
    # Черняхівське поселення:
    # не вершина і не крутий яр,
    # а відносно пологий схил.
    #

    if 2 <= slope <= 8:
        slope_score = 1.0

    elif slope < 2:
        slope_score = 0.7

    elif slope <= 12:
        slope_score = 0.6

    else:
        slope_score = 0.1

    # -----------------------------------------------------
    # SUN
    # -----------------------------------------------------

    if aspect is None:
        sun_score = 0.5

    else:

        # Південь = 180°
        # Південний схід = 135°
        # Південний захід = 225°

        distance = min(
            abs(aspect - 180),
            360 - abs(aspect - 180)
        )

        sun_score = max(
            0.0,
            1.0 - distance / 135
        )

    # -----------------------------------------------------
    # ПОКИ ЩО ТІЛЬКИ БАЗОВІ ОЗНАКИ
    # -----------------------------------------------------

    score = (
        CHERNYAKHIV_WEIGHTS["slope"]
        * slope_score
        +
        CHERNYAKHIV_WEIGHTS["sun"]
        * sun_score
    )

    # Переводимо в %
    return round(
        score * 100,
        1
    )


# =========================================================
# COPERNICUS AUTH
# =========================================================

def get_copernicus_token():
    """
    Отримання OAuth токена Copernicus Data Space.

    Ключі НЕ зберігаємо в коді.
    На Render вони будуть Environment Variables.
    """

    if not (
        COPERNICUS_CLIENT_ID
        and
        COPERNICUS_CLIENT_SECRET
    ):
        return None

    url = (
        "https://identity.dataspace.copernicus.eu/"
        "auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )

    data = {
        "grant_type": "client_credentials",
        "client_id": COPERNICUS_CLIENT_ID,
        "client_secret": COPERNICUS_CLIENT_SECRET
    }

    response = requests.post(
        url,
        data=data,
        timeout=30
    )

    response.raise_for_status()

    return response.json()[
        "access_token"
    ]


# =========================================================
# COPERNICUS SEARCH
# =========================================================

def search_sentinel2(
    lat,
    lon,
    radius_km=30,
    max_cloud=10
):
    """
    Пошук Sentinel-2 L2A сцен.

    Повертає метадані доступних знімків.

    Поки що НЕ завантажуємо весь знімок.
    """

    token = get_copernicus_token()

    if token is None:
        return {
            "enabled": False,
            "message": (
                "Copernicus credentials "
                "не налаштовані"
            )
        }

    bbox = get_bbox(
        lat,
        lon,
        radius_km
    )

    url = (
        "https://sh.dataspace.copernicus.eu/"
        "catalog/v1/search"
    )

    headers = {
        "Authorization":
            f"Bearer {token}",
        "Content-Type":
            "application/json"
    }

    payload = {

        "collections": [
            "sentinel-2-l2a"
        ],

        "datetime":
            "2026-04-01T00:00:00Z/"
            "2026-09-30T23:59:59Z",

        "bbox": [
            bbox["min_lon"],
            bbox["min_lat"],
            bbox["max_lon"],
            bbox["max_lat"]
        ],

        "limit": 20,

        "filter": {
            "op": "<=",
            "args": [
                {
                    "property":
                        "eo:cloud_cover"
                },
                max_cloud
            ]
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# API: STATUS
# =========================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": VERSION,
        "center": {
            "lat": CENTER_LAT,
            "lon": CENTER_LON
        },
        "radius_km":
            DEFAULT_RADIUS_KM
    }


# =========================================================
# API: TERRAIN
# =========================================================

@app.get("/api/analyze")
def analyze(
    lat: float = CENTER_LAT,
    lon: float = CENTER_LON,
    radius_km: float = DEFAULT_RADIUS_KM
):

    if radius_km <= 0:
        raise HTTPException(
            400,
            "radius_km має бути > 0"
        )

    if radius_km > 100:
        raise HTTPException(
            400,
            "Максимальний радіус зараз 100 км"
        )

    grid, lats, lons = load_dem(
        lat,
        lon,
        radius_km
    )

    rows = len(grid)
    cols = len(grid[0])

    # Приблизний розмір клітини
    cell_size_m = (
        radius_km * 2000
    ) / (rows - 1)

    results = []

    for r in range(
        1,
        rows - 1
    ):

        for c in range(
            1,
            cols - 1
        ):

            if grid[r][c] is None:
                continue

            slope = calculate_slope(
                grid,
                r,
                c,
                cell_size_m
            )

            aspect = calculate_aspect(
                grid,
                r,
                c,
                cell_size_m
            )

            score = (
                calculate_chernyakhiv_score(
                    slope,
                    aspect
                )
            )

            # Поки не віддаємо всі 32 000 точок
            if score >= 8:

                results.append({
                    "lat":
                        round(
                            lats[r],
                            5
                        ),

                    "lon":
                        round(
                            lons[c],
                            5
                        ),

                    "score":
                        score,

                    "elevation_m":
                        round(
                            grid[r][c],
                            1
                        ),

                    "slope_deg":
                        round(
                            slope,
                            1
                        )
                        if slope is not None
                        else None,

                    "aspect_deg":
                        round(
                            aspect,
                            1
                        )
                        if aspect is not None
                        else None
                })

    results.sort(
        key=lambda x:
            x["score"],
        reverse=True
    )

    return {
        "center": {
            "lat": lat,
            "lon": lon
        },

        "radius_km":
            radius_km,

        "profile":
            "chernyakhiv",

        "count":
            len(results),

        "results":
            results[:500]
    }


# =========================================================
# API: COPERNICUS
# =========================================================

@app.get("/api/sentinel")
def sentinel(
    lat: float = CENTER_LAT,
    lon: float = CENTER_LON,
    radius_km: float = DEFAULT_RADIUS_KM
):

    try:

        return search_sentinel2(
            lat,
            lon,
            radius_km,
            MAX_CLOUD_COVER
        )

    except Exception as error:

        raise HTTPException(
            500,
            f"Copernicus error: {error}"
        )


# =========================================================
# DEBUG MAP
# =========================================================

@app.get(
    "/map",
    response_class=HTMLResponse
)
def map_view(
    lat: float = CENTER_LAT,
    lon: float = CENTER_LON,
    radius_km: float = DEFAULT_RADIUS_KM
):

    # -----------------------------------------------------
    # Аналіз
    # -----------------------------------------------------

    data = analyze(
        lat,
        lon,
        radius_km
    )

    results = data["results"]

    # -----------------------------------------------------
    # MAP
    # -----------------------------------------------------

    m = folium.Map(
        location=[
            lat,
            lon
        ],
        zoom_start=11,
        tiles=None
    )

    # -----------------------------------------------------
    # SATELLITE
    # -----------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Супутник"
    ).add_to(m)

    # -----------------------------------------------------
    # TOPO
    # -----------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Топографічна карта"
    ).add_to(m)

    # -----------------------------------------------------
    # ANALYSIS AREA
    # -----------------------------------------------------

    folium.Circle(
        location=[
            lat,
            lon
        ],
        radius=radius_km * 1000,
        color="blue",
        fill=False,
        weight=2,
        popup=(
            f"Зона аналізу: "
            f"{radius_km} км"
        )
    ).add_to(m)

    # -----------------------------------------------------
    # CENTER
    # -----------------------------------------------------

    folium.Marker(
        [
            lat,
            lon
        ],
        popup="Центр аналізу — Ромни"
    ).add_to(m)

    # -----------------------------------------------------
    # CANDIDATES
    # -----------------------------------------------------

    for index, point in enumerate(
        results[:150],
        1
    ):

        score = point["score"]

        if score >= 35:
            color = "red"

        elif score >= 20:
            color = "orange"

        else:
            color = "blue"

        popup = f"""
        <b>Кандидат #{index}</b><br>
        Черняхівський score:
        {score}%<br>
        Висота:
        {point['elevation_m']} м<br>
        Схил:
        {point['slope_deg']}°<br>
        Експозиція:
        {point['aspect_deg']}°
        """

        folium.CircleMarker(
            location=[
                point["lat"],
                point["lon"]
            ],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                popup,
                max_width=250
            )
        ).add_to(m)

    # -----------------------------------------------------
    # LAYER CONTROL
    # -----------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m._repr_html_()
```
