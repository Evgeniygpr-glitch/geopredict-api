import os
import math
from functools import lru_cache

import numpy as np
import srtm
import folium
from folium.raster_layers import WmsTileLayer
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------------------------------
# Налаштування
# --------------------------------------------------------------------------

# На Render файлова система ефемерна, але /tmp доступний для запису
# протягом життя інстансу — SRTM-тайли кешуються туди, щоб не тягнути
# їх повторно при кожному холодному старті контейнера.
os.environ.setdefault("SRTM1_DIR", "/tmp/srtm_data")
os.environ.setdefault("SRTM3_DIR", "/tmp/srtm_data")
os.makedirs("/tmp/srtm_data", exist_ok=True)

MAX_RADIUS_KM = 20.0
MIN_RADIUS_KM = 0.5
TARGET_CELL_M = 35.0   # цільовий розмір клітинки сітки (SRTM ~30м/піксель)
MIN_GRID_STEPS = 40
MAX_GRID_STEPS = 160    # жорсткий стеля, щоб запит завжди встигав у таймаут Render

app = FastAPI(title="GeoPredict API (КР + WMS NDVI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # мобільний застосунок на Solar2D ходить крос-доменно
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Ліниве завантаження — щоб /  та /docs відповідали миттєво,
# навіть якщо мережа до джерела SRTM тимчасово недоступна.
_elevation_data = None


def get_elevation_source():
    global _elevation_data
    if _elevation_data is None:
        _elevation_data = srtm.get_data()
    return _elevation_data


# --------------------------------------------------------------------------
# Геоморфологія
# --------------------------------------------------------------------------

def calc_geomorphology(grid: np.ndarray, r: int, c: int, cell_size_m: float):
    dz_dx = (grid[r, c + 1] - grid[r, c - 1]) / (2 * cell_size_m)
    dz_dy = (grid[r + 1, c] - grid[r - 1, c]) / (2 * cell_size_m)
    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    aspect_deg = math.degrees(math.atan2(-dz_dy, dz_dx)) % 360

    aspect_rad = math.radians(aspect_deg)
    ideal_rad = math.radians(135.0)  # Південно-східний схил для КР
    sun_score = max(0.0, math.cos(aspect_rad - ideal_rad))
    return slope_deg, aspect_deg, round(sun_score, 2)


def detect_promontory_kr(grid: np.ndarray, r: int, c: int) -> float:
    """Детектор мисових форм рельєфу: перепад висоти з 3+ сторін."""
    z = float(grid[r, c])  # явний python float — інакше numpy.float32 "протікає" далі і ламає JSON-серіалізацію
    rows, cols = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    lower_count = 0
    max_drop = 0.0

    for dr, dc in dirs:
        for step in (1, 2, 3):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < rows and 0 <= nc < cols:
                drop = z - float(grid[nr, nc])
                if drop >= 3.5:
                    lower_count += 1
                    max_drop = max(max_drop, drop)
                    break

    if lower_count >= 3 and max_drop >= 4.0:
        return min(1.0, (lower_count / 8.0) * 0.5 + (max_drop / 15.0) * 0.5)
    return 0.0


def nearest_lower_point(grid: np.ndarray, r: int, c: int, search_r: int, cell_size_m: float):
    """
    Векторизований (numpy) пошук найнижчої точки в околі та відстані до неї.
    Замінює вкладені Python-цикли — на порядки швидше при великих сітках.
    """
    rows, cols = grid.shape
    r0, r1 = max(0, r - search_r), min(rows, r + search_r + 1)
    c0, c1 = max(0, c - search_r), min(cols, c + search_r + 1)

    sub = grid[r0:r1, c0:c1]
    min_idx = np.unravel_index(np.argmin(sub), sub.shape)
    min_z = float(sub[min_idx])

    nr, nc = r0 + min_idx[0], c0 + min_idx[1]
    dist_m = math.sqrt(((nr - r) * cell_size_m) ** 2 + ((nc - c) * cell_size_m) ** 2)
    return min_z, dist_m


def analyze_site_kr(grid: np.ndarray, r: int, c: int, lat_v: float, lon_v: float, cell_size_m: float):
    z_center = float(grid[r, c])

    tip_score = detect_promontory_kr(grid, r, c)
    if tip_score == 0.0:
        return None

    slope_deg, aspect_deg, sun_score = calc_geomorphology(grid, r, c, cell_size_m)

    search_r = max(3, min(10, int(450.0 / cell_size_m)))
    min_z, dist_to_water_m = nearest_lower_point(grid, r, c, search_r, cell_size_m)

    delta_h = z_center - min_z
    if delta_h < 6.0:
        return None

    if 10.0 <= delta_h <= 30.0:
        s_height = 1.0
    elif delta_h < 10.0:
        s_height = delta_h / 10.0
    else:
        s_height = max(0.4, 1.0 - (delta_h - 30.0) / 40.0)

    s_water = max(0.1, 1.0 - abs(dist_to_water_m - 200.0) / 350.0)

    final_score = (0.45 * tip_score + 0.25 * s_height + 0.20 * s_water + 0.10 * sun_score) * 100

    # Явні float()/bool()/int() — запобіжник від numpy-скалярів (numpy.float32/np.bool_
    # не серіалізуються в JSON і викликають 500 Internal Server Error), незалежно від
    # того, звідки саме вище по коду міг "протекти" numpy-тип.
    return {
        "lat": float(lat_v),
        "lon": float(lon_v),
        "score": round(float(final_score), 1),
        "elevation_m": round(float(z_center), 1),
        "delta_h_m": round(float(delta_h), 1),
        "dist_water_m": int(round(float(dist_to_water_m))),
        "slope_deg": round(float(slope_deg), 1),
        "aspect_deg": round(float(aspect_deg), 1),
        "sun_score": float(sun_score),
        "is_tip": bool(tip_score >= 0.65),
    }


def strict_nms_clustering(results: list, min_dist_m: float) -> list:
    results.sort(key=lambda x: x["score"], reverse=True)
    filtered = []
    for pt in results:
        keep = True
        for existing in filtered:
            d_lat = (pt["lat"] - existing["lat"]) * 111000
            d_lon = (pt["lon"] - existing["lon"]) * 111000 * math.cos(math.radians(pt["lat"]))
            dist = math.sqrt(d_lat ** 2 + d_lon ** 2)
            if dist < min_dist_m:
                keep = False
                break
        if keep:
            filtered.append(pt)
    return filtered


def build_grid(lat: float, lon: float, radius_km: float, grid_steps: int):
    """Будує сітку висот через numpy-масив, з одним викликом get_elevation на клітинку."""
    src = get_elevation_source()

    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

    lats = np.linspace(lat - lat_delta, lat + lat_delta, grid_steps + 1)
    lons = np.linspace(lon - lon_delta, lon + lon_delta, grid_steps + 1)
    cell_size_m = (2 * radius_km * 1000.0) / grid_steps

    grid = np.zeros((len(lats), len(lons)), dtype=np.float32)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            alt = src.get_elevation(float(la), float(lo))
            grid[i, j] = alt if alt is not None else 0.0

    return grid, lats, lons, cell_size_m


def choose_grid_steps(radius_km: float) -> int:
    steps = int((2 * radius_km * 1000.0) / TARGET_CELL_M)
    return max(MIN_GRID_STEPS, min(MAX_GRID_STEPS, steps))


@lru_cache(maxsize=64)
def run_analysis(lat: float, lon: float, radius_km: float) -> tuple:
    """Кешований аналіз: однакові (lat, lon, radius_km) рахуються лише один раз."""
    grid_steps = choose_grid_steps(radius_km)
    grid, lats, lons, cell_size_m = build_grid(lat, lon, radius_km, grid_steps)

    rows, cols = grid.shape
    raw_results = []

    for r in range(3, rows - 3):
        for c in range(3, cols - 3):
            lat_v = round(float(lats[r]), 5)
            lon_v = round(float(lons[c]), 5)
            res = analyze_site_kr(grid, r, c, lat_v, lon_v, cell_size_m)
            if res and res["score"] >= 42.0:
                raw_results.append(res)

    clean_results = strict_nms_clustering(raw_results, min_dist_m=130.0)
    limit = 35 if radius_km >= 10 else 20
    return tuple(clean_results[:limit])


# --------------------------------------------------------------------------
# Ендпоінти
# --------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "GeoPredict API (КР + WMS NDVI) працює"}


@app.get("/api/analyze")
def api_analyze(
    lat: float = Query(50.75, ge=-85, le=85),
    lon: float = Query(33.47, ge=-180, le=180),
    radius_km: float = Query(10.0, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM),
):
    """
    Легкий JSON-ендпоінт для мобільного клієнта (Solar2D).
    Повертає лише координати та метрики знайдених об'єктів, без HTML.
    """
    results = run_analysis(round(lat, 5), round(lon, 5), round(radius_km, 2))
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "count": len(results),
        "results": list(results),
    }


@app.get("/map", response_class=HTMLResponse)
def get_map(
    lat: float = Query(50.75, ge=-85, le=85),
    lon: float = Query(33.47, ge=-180, le=180),
    radius_km: float = Query(10.0, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM),
):
    """HTML-мапа лишається для власного дебагу в браузері — Solar2D її не використовує."""
    results = run_analysis(round(lat, 5), round(lon, 5), round(radius_km, 2))

    m = folium.Map(
        location=[lat, lon],
        zoom_start=12 if radius_km > 10 else 14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🌍 Супутник HD (Esri)",
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Maps",
        name="🛰️ Google Hybrid (з назвами)",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Topo",
        name="🗺️ Топо-карта (Esri Topo)",
    ).add_to(m)

    WmsTileLayer(
        url="https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
        layers="MODIS_Terra_NDVI_8Day",
        name="🌱 NDVI Індекс рослинності (NASA)",
        fmt="image/png",
        transparent=True,
        overlay=True,
        opacity=0.6,
        attr="NASA GIBS / EOSDIS",
    ).add_to(m)

    WmsTileLayer(
        url="https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
        layers="MODIS_Terra_CorrectedReflectance_Bands721",
        name="🌾 Контраст полів (False Color 7-2-1)",
        fmt="image/jpeg",
        transparent=False,
        overlay=True,
        opacity=0.65,
        attr="NASA GIBS / EOSDIS",
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=radius_km * 1000,
        color="#e74c3c",
        weight=2,
        fill=True,
        fill_color="#e74c3c",
        fill_opacity=0.08,
        popup=f"Зона аналізу: {radius_km} км",
        tooltip=f"Радіус аналізу {radius_km} км",
    ).add_to(m)

    folium.Marker(
        [lat, lon],
        popup=f"Центр аналізу КР ({lat:.5f}, {lon:.5f})",
        icon=folium.Icon(color="black", icon="info-sign"),
    ).add_to(m)

    for idx, pt in enumerate(results, 1):
        if pt["score"] >= 68:
            color = "red"
        elif pt["score"] >= 52:
            color = "orange"
        else:
            color = "darkblue"

        popup_html = f"""
        <div style='font-family: sans-serif; width: 230px;'>
            <h4 style='margin:0 0 5px 0; color:#d9534f;'>Ціль КР #{idx} (Бал: {pt['score']}%)</h4>
            <b>Тип:</b> {'Оборонний мис (Городище)' if pt['is_tip'] else 'Терасове селище'}<br>
            <b>Висота над низиною:</b> +{pt['delta_h_m']} м<br>
            <b>Абс. висота:</b> {pt['elevation_m']} м<br>
            <b>До річки/заплави:</b> ~{pt['dist_water_m']} м<br>
            <b>Схил / Сонце:</b> {pt['aspect_deg']}° ({int(pt['sun_score'] * 100)}%)
        </div>
        """

        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=7 if radius_km > 10 else 9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()
