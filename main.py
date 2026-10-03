import os
import re
import math
import io
import json
import time
import uuid
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import requests
from PIL import Image
import srtm
import folium
from folium.raster_layers import WmsTileLayer, ImageOverlay
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response, JSONResponse
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

MAX_RADIUS_KM = 50.0
MIN_RADIUS_KM = 0.5
TARGET_CELL_M = 35.0   # цільовий розмір клітинки сітки (SRTM ~30м/піксель)
MIN_GRID_STEPS = 40
# Було 160 -> при радіусі 10 км клітинка ~125 м, мисів/терас просто не видно.
# Тепер розрахунок іде у фоновому потоці з прогрес-баром (/loading), тож стелю можна підняти.
MAX_GRID_STEPS = int(os.environ.get("MAX_GRID_STEPS", "300"))

# --- Параметри "скільки точок показувати" (можна міняти через змінні середовища Render) ---
MIN_SCORE = float(os.environ.get("MIN_SCORE", "36"))             # було 42
MAX_POINTS_BIG = int(os.environ.get("MAX_POINTS_BIG", "80"))     # радіус >= 10 км, було 35
MAX_POINTS_SMALL = int(os.environ.get("MAX_POINTS_SMALL", "50")) # радіус < 10 км, було 20
NMS_KR_M = float(os.environ.get("NMS_KR_M", "90"))               # було 130
NMS_CHERN_M = float(os.environ.get("NMS_CHERN_M", "70"))         # було 100
KR_DROP_MIN = float(os.environ.get("KR_DROP_MIN", "3.0"))        # було 3.5 м
KR_MAX_DROP_MIN = float(os.environ.get("KR_MAX_DROP_MIN", "3.5"))  # було 4.0 м
KR_DELTA_H_MIN = float(os.environ.get("KR_DELTA_H_MIN", "5.0"))  # було 6.0 м

# --- Прогрес довгих розрахунків ---
_tl = threading.local()
_PHASES = {
    "grid":    (0.00, 0.45, "Висоти SRTM"),
    "ravines": (0.45, 0.55, "Яри та джерела"),
    "hollows": (0.55, 0.65, "Ложбини"),
    "scan":    (0.65, 0.93, "Пошук точок"),
    "harvest": (0.93, 1.00, "Супутникові знімки NDVI"),
}


def _report(phase: str, frac: float = 0.0, detail: str = ""):
    """Повідомляє прогрес, якщо поточний потік запущено як фонове завдання."""
    cb = getattr(_tl, "cb", None)
    if cb is not None:
        cb(phase, frac, detail)

# --------------------------------------------------------------------------
# Прекомп'ютовані яри/джерела (щоб не рахувати наживо щоразу)
# --------------------------------------------------------------------------
# Одноразово прораховуєте великий регіон через /api/ravines?...&download=true,
# зберігаєте отриманий JSON у репозиторій як data/precomputed_ravines.json,
# і при наступних запитах для тієї ж зони дані читаються миттєво з файлу
# замість повторного SRTM-аналізу.
PRECOMPUTED_RAVINES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "precomputed_ravines.json")
PRECOMPUTED_TILES = []

try:
    with open(PRECOMPUTED_RAVINES_PATH, "r", encoding="utf-8") as f:
        PRECOMPUTED_TILES = json.load(f).get("tiles", [])
    print(f"[precompute] Завантажено {len(PRECOMPUTED_TILES)} прекомп'ютованих плиток ярів.")
except FileNotFoundError:
    print(f"[precompute] Файл {PRECOMPUTED_RAVINES_PATH} не знайдено — працюємо в режимі живого обчислення.")
except Exception as exc:
    print(f"[precompute] Помилка читання {PRECOMPUTED_RAVINES_PATH}: {exc}")


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = (lat1 - lat2) * 111000.0
    d_lon = (lon1 - lon2) * 111000.0 * math.cos(math.radians(lat1))
    return math.sqrt(d_lat ** 2 + d_lon ** 2)


def _bbox(lat: float, lon: float, radius_km: float):
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta


def find_covering_tile(lat: float, lon: float, radius_km: float):
    """Чи є прекомп'ютована плитка, яка ПОВНІСТЮ покриває запитувану область?"""
    q_min_lat, q_max_lat, q_min_lon, q_max_lon = _bbox(lat, lon, radius_km)
    for tile in PRECOMPUTED_TILES:
        b = tile["bounds"]
        if (b["min_lat"] <= q_min_lat and b["max_lat"] >= q_max_lat and
                b["min_lon"] <= q_min_lon and b["max_lon"] >= q_max_lon):
            return tile
    return None

# Опційна інтеграція Copernicus Data Space (Sentinel Hub) для чіткого NDVI (10 м/піксель).
# Якщо змінна не задана — використовується запасний шар NASA GIBS MODIS NDVI (250-500 м/піксель,
# помітно грубіший, зате не вимагає реєстрації).
COPERNICUS_INSTANCE_ID = os.environ.get("COPERNICUS_INSTANCE_ID", "").strip()

# ID вашого кастомного evalscript-шару (створюється вручну в Configuration Utility —
# див. інструкцію в чаті). За замовчуванням очікується назва MOWN_DETECT, але можна
# перейменувати через змінну середовища, якщо назвали шар інакше.
COPERNICUS_MOWN_LAYER = os.environ.get("COPERNICUS_MOWN_LAYER", "HARVESTED-FIELDS").strip()
COPERNICUS_NDVI_RAW_LAYER = os.environ.get("COPERNICUS_NDVI_RAW_LAYER", "NDVI_RAW").strip()

# Розмір боку зображення для порівняння пік-вегетації/поточного стану (px).
# Обмежено заради швидкості й розміру відповіді Render.
HARVEST_OVERLAY_MAX_PX = 500

# Вікна (днів тому) для знімків: перше — поточний стан, решта — історія для пошуку піку вегетації.
HARVEST_SNAPSHOT_WINDOWS_DAYS = [
    (0, 30),      # поточний стан
    (45, 75),
    (90, 120),
    (135, 165),   # ширше вікно — шанс захопити пік вегетації перед збором урожаю
]

# Скільки днів назад шукати безхмарний знімок. Sentinel-2 пролітає над однією точкою
# приблизно раз на 5 днів — 60-денне вікно майже завжди дає хоча б один прийнятний кадр,
# а WMS сам обере найсвіжіший (PRIORITY=mostRecent) серед тих, де хмарність <= MAX_CLOUD_PCT.
CLOUD_SEARCH_WINDOW_DAYS = 60
HARVEST_DETECT_WINDOW_DAYS = 150  # ширше вікно — треба захопити пік вегетації ДО збору врожаю
MAX_CLOUD_PCT = 20


def sentinel_time_range(days: int = CLOUD_SEARCH_WINDOW_DAYS) -> str:
    """Діапазон часу для WMS TIME-параметра: останні `days` днів."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return f"{start.isoformat()}/{end.isoformat()}"


_NDVI_LAYER_MISSING = {"flag": False}


def _fetch_ndvi_snapshot(sh_url: str, bbox, width: int, height: int, days_from: int, days_to: int):
    """
    Один WMS GetMap запит до шару NDVI_RAW за вікно [days_to; days_from] днів тому.
    Повертає (ndvi_array, valid_mask) або None, якщо запит не вдався чи немає безхмарного знімка.
    """
    end = datetime.now(timezone.utc).date() - timedelta(days=days_from)
    start = datetime.now(timezone.utc).date() - timedelta(days=days_to)
    time_range = f"{start.isoformat()}/{end.isoformat()}"

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": COPERNICUS_NDVI_RAW_LAYER,
        "SRS": "EPSG:4326",
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
        "TIME": time_range,
        "PRIORITY": "mostRecent",
        "MAXCC": 30,
    }

    try:
        resp = requests.get(sh_url, params=params, timeout=25)
        if resp.status_code != 200:
            # Показуємо тіло відповіді — Sentinel Hub зазвичай пише точну причину помилки в XML.
            print(f"[harvest_overlay] HTTP {resp.status_code} від Copernicus: {resp.text[:500]}")
            if "LayerNotDefined" in resp.text:
                _NDVI_LAYER_MISSING["flag"] = True
            resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("LA")
        arr = np.array(img)
        gray = arr[..., 0].astype(np.float32)
        alpha = arr[..., 1]
        ndvi = gray / 127.5 - 1.0
        valid = alpha > 200
        return ndvi, valid
    except Exception as exc:
        print(f"[harvest_overlay] Не вдалось отримати знімок ({days_from}-{days_to} днів тому): {exc}")
        return None


@lru_cache(maxsize=16)
def build_harvest_overlay(lat: float, lon: float, radius_km: float):
    """
    Порівнює пік вегетації (за останні ~5.5 міс) з поточним станом і повертає
    RGBA numpy-масив, де підсвічено ТІЛЬКИ ділянки з різким падінням NDVI —
    типова ознака недавнього збору врожаю/скошування. Гола земля, яка й раніше
    була голою (низький NDVI в усіх знімках), під умову не потрапляє.
    Повертає None, якщо дані недоступні (немає Instance ID, мережева помилка,
    немає жодного придатного безхмарного знімка).
    """
    if not COPERNICUS_INSTANCE_ID:
        return None

    sh_url = f"https://sh.dataspace.copernicus.eu/ogc/wms/{COPERNICUS_INSTANCE_ID}"

    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    bbox = (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)  # minx,miny,maxx,maxy

    side_m = 2 * radius_km * 1000.0
    px = max(200, min(HARVEST_OVERLAY_MAX_PX, int(side_m / 25)))

    windows = HARVEST_SNAPSHOT_WINDOWS_DAYS
    _report("harvest", 0.0, f"Знімок 1 з {len(windows)} (поточний стан)")
    first = _fetch_ndvi_snapshot(sh_url, bbox, px, px, *windows[0])
    if _NDVI_LAYER_MISSING["flag"]:
        print(f"[harvest_overlay] Шар '{COPERNICUS_NDVI_RAW_LAYER}' не створено в Sentinel Hub "
              f"Configuration Utility — решту запитів пропускаю.")
        return None

    from concurrent.futures import ThreadPoolExecutor
    rest = windows[1:]
    _report("harvest", 0.3, f"Історичні знімки ({len(rest)} шт.) — паралельно")
    with ThreadPoolExecutor(max_workers=len(rest)) as pool:
        futures = [pool.submit(_fetch_ndvi_snapshot, sh_url, bbox, px, px, df, dt) for df, dt in rest]
        others = [f.result() for f in futures]
    snapshots = [first] + others

    current = snapshots[0]
    history = [s for s in snapshots[1:] if s is not None]

    if current is None:
        print("[harvest_overlay] Немає поточного (0-30 днів) знімка — пропускаю шар.")
        return None
    if not history:
        print("[harvest_overlay] Немає жодного історичного знімка для порівняння — пропускаю шар.")
        return None

    current_ndvi, current_valid = current

    peak_ndvi = np.full_like(current_ndvi, -1.0)
    peak_valid = np.zeros_like(current_valid)
    for ndvi, valid in history:
        take = valid & (ndvi > peak_ndvi)
        peak_ndvi = np.where(take, ndvi, peak_ndvi)
        peak_valid = peak_valid | valid

    harvested = (
        current_valid & peak_valid &
        (peak_ndvi > 0.4) &          # раніше там точно щось росло
        (current_ndvi < 0.3) &       # зараз низька рослинність
        ((peak_ndvi - current_ndvi) > 0.2)  # суттєве падіння
    )

    h, w = harvested.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[harvested] = [255, 140, 0, 200]  # помаранчевий, напівпрозорий; решта лишається прозорою
    return overlay

app = FastAPI(title="GeoPredict API (КР + WMS NDVI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # мобільний застосунок на Solar2D ходить крос-доменно
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Публічна адреса сервісу — для посилань усередині карти (вона рендериться в iframe,
# тому відносні посилання не працюють). Можна перекрити змінною PUBLIC_BASE_URL.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://geopredict-api.onrender.com").rstrip("/")

# AI-дослідження зони (окремий файл research.py). Якщо файлу немає на GitHub —
# решта застосунку працює як раніше, просто без цього розділу.
try:
    from research import router as research_router
    app.include_router(research_router)
    print("[research] Модуль AI-дослідження підключено.")
except Exception as exc:
    print(f"[research] Модуль не підключено: {exc}")

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
                if drop >= KR_DROP_MIN:
                    lower_count += 1
                    max_drop = max(max_drop, drop)
                    break

    if lower_count >= 3 and max_drop >= KR_MAX_DROP_MIN:
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
    if delta_h < KR_DELTA_H_MIN:
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
        "culture": "kr",
    }


def score_height_chernyakhiv(delta_h: float) -> float:
    """Фактор А: висота над заплавою. Ідеал 3-10 м, різкий спад за межами."""
    if delta_h < 2.0 or delta_h > 25.0:
        return 0.0
    if delta_h < 3.0:
        return delta_h / 3.0  # лінійний перехід 2.0м(=0) -> 3.0м(=1.0)
    if delta_h <= 10.0:
        return 1.0
    # 10-25 м: лінійне падіння до нуля
    return max(0.0, 1.0 - (delta_h - 10.0) / 15.0)


def score_water_chernyakhiv(dist_water_m: float) -> float:
    """Фактор Б: відстань до води. Ідеал <=150 м, далі спадає."""
    if dist_water_m <= 150.0:
        return 1.0
    return max(0.0, 1.0 - (dist_water_m - 150.0) / 350.0)


def score_slope_chernyakhiv(slope_deg: float) -> float:
    """Фактор В: крутизна. Оптимум 2-6°, надто рівно чи надто круто — гірше."""
    if 2.0 <= slope_deg <= 6.0:
        return 1.0
    if slope_deg < 2.0:
        return 0.5 + 0.5 * (slope_deg / 2.0)  # 0°→0.5, 2°→1.0 (погано дренує, але не критично)
    # 6-10°: спад до нуля (>10° відсікається раніше жорстким фільтром плато)
    return max(0.0, 1.0 - (slope_deg - 6.0) / 4.0)


def score_sun_chernyakhiv(aspect_deg: float) -> float:
    """Фактор Г: експозиція. Ідеал — Південь (180°), прийнятно ПдС/ПдЗ."""
    ideal_rad = math.radians(180.0)
    aspect_rad = math.radians(aspect_deg)
    return max(0.0, math.cos(aspect_rad - ideal_rad))


def score_spring_chernyakhiv(dist_spring_m: float) -> float:
    """
    Фактор Д: відстань до найближчого ЯВНОГО початку яру/джерела (з D8-аналізу стоку).
    Черняхівці охоче селились саме біля витоку струмка — там чиста джерельна вода.
    """
    if dist_spring_m is None or not math.isfinite(dist_spring_m):
        return 0.0
    if dist_spring_m <= 100.0:
        return 1.0
    return max(0.0, 1.0 - (dist_spring_m - 100.0) / 300.0)  # спадає до 0 приблизно на 400м


def score_hollow_chernyakhiv(dist_hollow_m: float) -> float:
    """
    Фактор Е: відстань до найближчої придатної ложбини під напівземлянку
    (знайденої окремим прицільним пошуком біля витоків яруг — див.
    find_ravine_head_hollows). Чим ближче до такої ложбини — тим ймовірніше
    саме тут копали напівземлянку.
    """
    if dist_hollow_m is None or not math.isfinite(dist_hollow_m):
        return 0.0
    if dist_hollow_m <= 60.0:
        return 1.0
    return max(0.0, 1.0 - (dist_hollow_m - 60.0) / 200.0)  # спадає до 0 приблизно на 260м


def analyze_site_chernyakhiv(grid: np.ndarray, r: int, c: int, lat_v: float, lon_v: float,
                              cell_size_m: float, dist_to_spring_m: float = None,
                              dist_to_hollow_m: float = None):
    z_center = float(grid[r, c])

    slope_deg, aspect_deg, _ = calc_geomorphology(grid, r, c, cell_size_m)

    # Фільтр плато: черняхівці не будували на схилах крутіших за 10° —
    # і, на відміну від КР, тут НЕ шукаємо мисовий перепад з 3+ сторін:
    # тераса може плавно переходити у височину з одного боку.
    if slope_deg > 10.0:
        return None

    search_r = max(3, min(12, int(450.0 / cell_size_m)))
    min_z, dist_to_water_m = nearest_lower_point(grid, r, c, search_r, cell_size_m)
    delta_h = z_center - min_z

    s_height = score_height_chernyakhiv(delta_h)
    if s_height <= 0.0:
        return None  # затоплення (< 2м) або зависоко (> 25м) — категорично ні

    s_water = score_water_chernyakhiv(dist_to_water_m)
    s_slope = score_slope_chernyakhiv(slope_deg)
    s_sun = score_sun_chernyakhiv(aspect_deg)
    s_spring = score_spring_chernyakhiv(dist_to_spring_m)
    s_hollow = score_hollow_chernyakhiv(dist_to_hollow_m)

    # Висота над заплавою і вода — найкритичніші; близькість до витоку струмка
    # (джерела) і до придатної ложбини під напівземлянку — суттєві бонуси;
    # схил і сонце — допоміжні фактори.
    final_score = (
        0.25 * s_height +
        0.20 * s_water +
        0.15 * s_spring +
        0.20 * s_hollow +
        0.12 * s_slope +
        0.08 * s_sun
    ) * 100

    return {
        "lat": float(lat_v),
        "lon": float(lon_v),
        "score": round(float(final_score), 1),
        "elevation_m": round(float(z_center), 1),
        "delta_h_m": round(float(delta_h), 1),
        "dist_water_m": int(round(float(dist_to_water_m))),
        "dist_spring_m": int(round(float(dist_to_spring_m))) if dist_to_spring_m is not None and math.isfinite(dist_to_spring_m) else None,
        "dist_hollow_m": int(round(float(dist_to_hollow_m))) if dist_to_hollow_m is not None and math.isfinite(dist_to_hollow_m) else None,
        "slope_deg": round(float(slope_deg), 1),
        "aspect_deg": round(float(aspect_deg), 1),
        "sun_score": round(float(s_sun), 2),
        "culture": "cherniakhiv",
    }


def strict_nms_clustering(results: list, min_dist_m: float) -> list:
    """Non-max suppression: лишає найкращі точки, викидає сусідні ближче за min_dist_m.
    Просторові кошики замість повного перебору — на десятках тисяч кандидатів це секунди замість хвилин."""
    results.sort(key=lambda x: x["score"], reverse=True)
    if not results:
        return []
    kx = 111000.0 * math.cos(math.radians(results[0]["lat"]))
    ky = 111000.0
    min2 = min_dist_m * min_dist_m
    buckets = {}
    filtered = []
    for pt in results:
        x, y = pt["lon"] * kx, pt["lat"] * ky
        bx, by = int(x // min_dist_m), int(y // min_dist_m)
        keep = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ex, ey in buckets.get((bx + dx, by + dy), ()):
                    if (x - ex) ** 2 + (y - ey) ** 2 < min2:
                        keep = False
                        break
                if not keep:
                    break
            if not keep:
                break
        if keep:
            filtered.append(pt)
            buckets.setdefault((bx, by), []).append((x, y))
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
    n_rows = len(lats)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            alt = src.get_elevation(float(la), float(lo))
            grid[i, j] = alt if alt is not None else 0.0
        _report("grid", (i + 1) / n_rows, f"Рядок сітки {i + 1} з {n_rows}")

    return grid, lats, lons, cell_size_m


def choose_grid_steps(radius_km: float) -> int:
    steps = int((2 * radius_km * 1000.0) / TARGET_CELL_M)
    return max(MIN_GRID_STEPS, min(MAX_GRID_STEPS, steps))


@lru_cache(maxsize=32)
def get_grid_bundle(lat: float, lon: float, radius_km: float):
    """
    Спільна кешована сітка висот — і пошук городищ/терас, і аналіз ярів
    використовують ОДНУ й ту саму сітку замість повторних SRTM-запитів.
    """
    grid_steps = choose_grid_steps(radius_km)
    return build_grid(lat, lon, radius_km, grid_steps)


# --------------------------------------------------------------------------
# Аналіз ярів/струмків (D8 flow accumulation)
# --------------------------------------------------------------------------
# Стандартний гідрологічний метод: для кожної клітинки визначаємо, куди
# фізично стікала б вода (до найкрутішого нижчого сусіда — 8 напрямків),
# а тоді накопичуємо площу водозбору вниз за течією. Клітинки з великою
# накопиченою площею — це тальвеги яруг/струмків, навіть якщо зараз там
# сухо. "Голова" такого тальвега (де накопичена площа щойно перевищила
# поріг) — ймовірне місце виходу джерела, саме туди, за спостереженням,
# і селились черняхівці.

RAVINE_AREA_THRESHOLD_M2 = 20000.0  # ~2 га водозбору — поріг зарахування до "яру/струмка"
RAVINE_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def compute_flow_accumulation(grid: np.ndarray, cell_size_m: float):
    rows, cols = grid.shape
    flow_to = np.full((rows, cols, 2), -1, dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            z = grid[r, c]
            best_slope = 0.0
            best_rc = (-1, -1)
            for dr, dc in RAVINE_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    dist = cell_size_m * (1.41421356 if dr != 0 and dc != 0 else 1.0)
                    slope = (z - grid[nr, nc]) / dist
                    if slope > best_slope:
                        best_slope = slope
                        best_rc = (nr, nc)
            flow_to[r, c] = best_rc

    # Топологічна обробка від найвищих клітинок до найнижчих — накопичуємо площу вниз за течією.
    acc = np.ones((rows, cols), dtype=np.float64)
    order = np.argsort(-grid, axis=None)
    for idx in order:
        r, c = divmod(int(idx), cols)
        tr, tc = flow_to[r, c]
        if tr >= 0:
            acc[tr, tc] += acc[r, c]

    return acc, flow_to


def detect_channel_heads(grid: np.ndarray, acc: np.ndarray, flow_to: np.ndarray, cell_size_m: float):
    rows, cols = grid.shape
    cell_area_m2 = cell_size_m ** 2
    channel_mask = (acc * cell_area_m2) >= RAVINE_AREA_THRESHOLD_M2

    has_channel_contributor = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if not channel_mask[r, c]:
                continue
            for dr, dc in RAVINE_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    tr, tc = flow_to[nr, nc]
                    if tr == r and tc == c and channel_mask[nr, nc]:
                        has_channel_contributor[r, c] = True
                        break

    heads_mask = channel_mask & (~has_channel_contributor)
    return channel_mask, heads_mask


@lru_cache(maxsize=32)
@lru_cache(maxsize=8)
def get_channel_heads_and_masks(lat: float, lon: float, radius_km: float):
    """
    ЖИВЕ обчислення (без прекомп'ютованого кешу) яруг/джерел для заданого
    центру й радіуса. Повертає індекси голів яруг (r,c) у координатах сітки
    + маски. Викликається лише коли область НЕ покрита прекомп'ютованою плиткою.
    """
    grid, lats, lons, cell_size_m = get_grid_bundle(lat, lon, radius_km)
    acc, flow_to = compute_flow_accumulation(grid, cell_size_m)
    channel_mask, heads_mask = detect_channel_heads(grid, acc, flow_to, cell_size_m)
    head_indices = tuple(map(tuple, np.argwhere(heads_mask)))
    return head_indices, channel_mask, flow_to


def build_distance_grid(grid: np.ndarray, target_indices: tuple, cell_size_m: float) -> np.ndarray:
    """Для кожної клітинки сітки — відстань (м) до найближчої цільової точки (індекси r,c)."""
    rows, cols = grid.shape
    dist_grid = np.full((rows, cols), np.inf, dtype=np.float64)
    if not target_indices:
        return dist_grid

    # Швидкий шлях: одна EDT-операція замість циклу "голова яру x вся сітка".
    try:
        from scipy.ndimage import distance_transform_edt
        mask = np.ones((rows, cols), dtype=bool)
        for tr, tc in target_indices:
            mask[tr, tc] = False
        return distance_transform_edt(mask) * cell_size_m
    except ImportError:
        pass  # scipy нема в requirements.txt — працює повільніший запасний цикл нижче

    rr, cc = np.indices((rows, cols))
    for tr, tc in target_indices:
        d = np.sqrt((rr - tr) ** 2 + (cc - tc) ** 2) * cell_size_m
        dist_grid = np.minimum(dist_grid, d)
    return dist_grid


def get_channel_head_indices_for_scoring(lat: float, lon: float, radius_km: float,
                                          grid: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """
    Індекси голів яруг у координатах ПОТОЧНОЇ сітки — для побудови карти
    відстаней у run_analysis. Спершу перевіряє прекомп'ютований кеш
    (миттєво), і лише якщо область не покрита — рахує наживо (SRTM+D8).
    """
    tile = find_covering_tile(lat, lon, radius_km)
    if tile is not None:
        indices = []
        for h in tile["heads"]:
            if _dist_m(lat, lon, h["lat"], h["lon"]) <= radius_km * 1000.0:
                r = int(np.argmin(np.abs(lats - h["lat"])))
                c = int(np.argmin(np.abs(lons - h["lon"])))
                indices.append((r, c))
        return tuple(indices)

    head_indices, _, _ = get_channel_heads_and_masks(lat, lon, radius_km)
    return head_indices


@lru_cache(maxsize=32)
def _live_analyze_ravines(lat: float, lon: float, radius_km: float):
    """Повертає (heads, segments) ЖИВИМ обчисленням — використовується лише як fallback."""
    grid, lats, lons, cell_size_m = get_grid_bundle(lat, lon, radius_km)
    head_indices, channel_mask, flow_to = get_channel_heads_and_masks(lat, lon, radius_km)

    rows, cols = grid.shape
    heads = [
        {
            "lat": round(float(lats[r]), 5),
            "lon": round(float(lons[c]), 5),
            "elevation_m": round(float(grid[r, c]), 1),
        }
        for r, c in head_indices
    ]

    segments = []
    for r in range(rows):
        for c in range(cols):
            if channel_mask[r, c]:
                tr, tc = flow_to[r, c]
                if tr >= 0 and channel_mask[tr, tc]:
                    segments.append((
                        (round(float(lats[r]), 5), round(float(lons[c]), 5)),
                        (round(float(lats[tr]), 5), round(float(lons[tc]), 5)),
                    ))

    return tuple(heads), tuple(segments)


def get_ravines_for_area(lat: float, lon: float, radius_km: float):
    """
    Публічна точка входу для отримання ярів/джерел: спершу перевіряє
    прекомп'ютований кеш (миттєво, без SRTM-запитів), і лише якщо область
    НЕ покрита жодною збереженою плиткою — рахує наживо.
    """
    tile = find_covering_tile(lat, lon, radius_km)
    if tile is not None:
        radius_m = radius_km * 1000.0
        heads = tuple(h for h in tile["heads"] if _dist_m(lat, lon, h["lat"], h["lon"]) <= radius_m)
        segments = tuple(
            tuple(map(tuple, s)) for s in tile["segments"]
            if _dist_m(lat, lon, s[0][0], s[0][1]) <= radius_m or _dist_m(lat, lon, s[1][0], s[1][1]) <= radius_m
        )
        return heads, segments

    return _live_analyze_ravines(lat, lon, radius_km)


# --------------------------------------------------------------------------
# Масове прекомп'ютування всієї Сумської області ОДНІЄЮ кнопкою
# --------------------------------------------------------------------------
# Рахує всі 9 плиток (50 км кожна, з перекриттям) у ФОНОВОМУ потоці — тому
# HTTP-запит не висне і не впирається в таймаут Render, скільки б це не
# тривало. Сторінка /precompute сама опитує прогрес і показує посилання
# на завантаження готового ОДНОГО файлу, коли все готово.

SUMY_OBLAST_TILES = [
    (50.225, 32.958), (50.225, 34.032), (50.225, 35.105),
    (50.901, 32.958), (50.901, 34.032), (50.901, 35.105),
    (51.577, 32.958), (51.577, 34.032), (51.577, 35.105),
]
PRECOMPUTE_TILE_RADIUS_KM = 50.0

_precompute_job = {"status": "idle", "stage": "", "tiles": [], "error": None}
_precompute_lock = threading.Lock()


def _run_precompute_all():
    global _precompute_job
    try:
        tiles_out = []
        total = len(SUMY_OBLAST_TILES)
        for i, (t_lat, t_lon) in enumerate(SUMY_OBLAST_TILES, 1):
            with _precompute_lock:
                _precompute_job["stage"] = f"Плитка {i}/{total} ({t_lat}, {t_lon})… це повільно, чекайте"
            heads, segments = _live_analyze_ravines(t_lat, t_lon, PRECOMPUTE_TILE_RADIUS_KM)
            min_lat, max_lat, min_lon, max_lon = _bbox(t_lat, t_lon, PRECOMPUTE_TILE_RADIUS_KM)
            tiles_out.append({
                "center": {"lat": t_lat, "lon": t_lon},
                "radius_km": PRECOMPUTE_TILE_RADIUS_KM,
                "bounds": {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon},
                "heads": list(heads),
                "segments": [list(s) for s in segments],
            })
            with _precompute_lock:
                _precompute_job["tiles"] = tiles_out  # проміжний прогрес теж зберігаємо

        with _precompute_lock:
            _precompute_job["status"] = "done"
            _precompute_job["stage"] = "Готово! Натисніть 'Завантажити файл'."
    except Exception as exc:
        print(f"[precompute_all] Помилка: {exc}")
        with _precompute_lock:
            _precompute_job["status"] = "error"
            _precompute_job["error"] = str(exc)


@app.get("/api/precompute_all/start")
def precompute_all_start():
    with _precompute_lock:
        if _precompute_job["status"] == "running":
            return {"status": "running", "stage": _precompute_job["stage"]}
        _precompute_job.update(status="running", stage="Запуск…", tiles=[], error=None)
    threading.Thread(target=_run_precompute_all, daemon=True).start()
    return {"status": "running"}


@app.get("/api/precompute_all/status")
def precompute_all_status():
    with _precompute_lock:
        return {
            "status": _precompute_job["status"],
            "stage": _precompute_job["stage"],
            "error": _precompute_job["error"],
            "tiles_done": len(_precompute_job["tiles"]),
            "tiles_total": len(SUMY_OBLAST_TILES),
        }


@app.get("/api/precompute_all/download")
def precompute_all_download():
    with _precompute_lock:
        if _precompute_job["status"] != "done":
            return JSONResponse({"detail": "Ще не готово, зачекайте завершення."}, status_code=409)
        payload = {"tiles": _precompute_job["tiles"]}

    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=precomputed_ravines.json"},
    )


PRECOMPUTE_PAGE_HTML = """<!DOCTYPE html>
<html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Прекомп'ютування всієї області</title>
<style>
body{font-family:system-ui,sans-serif;max-width:600px;margin:0 auto;padding:16px;line-height:1.5}
button{padding:14px 18px;font-size:17px;width:100%;margin:8px 0}
#status{padding:12px;border-radius:8px;background:rgba(127,127,127,.15);margin-top:10px}
#status.err{background:rgba(220,50,50,.2)}
#dl{display:none;text-align:center;padding:14px;background:#2f9e44;color:#fff;
    border-radius:8px;text-decoration:none;font-weight:bold;margin-top:10px}
</style></head><body>
<h2>🗺️ Прекомп'ютування всієї Сумської області</h2>
<p>Порахує всі 9 плиток (по 50 км) одним натисканням. Це триває кілька хвилин —
можна закрити екран, прогрес зберігається на сервері.</p>
<button onclick="start()">▶️ Почати обчислення</button>
<div id="status">Натисніть кнопку вище.</div>
<a id="dl" href="/api/precompute_all/download">⬇️ Завантажити готовий файл precomputed_ravines.json</a>

<script>
const st = document.getElementById('status');
const dl = document.getElementById('dl');

async function start(){
  st.className = ''; dl.style.display = 'none';
  st.textContent = 'Запускаю…';
  await fetch('/api/precompute_all/start');
  poll();
}

async function poll(){
  for (let i = 0; i < 400; i++) {
    let s;
    try { s = await (await fetch('/api/precompute_all/status')).json(); }
    catch(e){ st.textContent = 'Втрачено звʼязок, пробую ще…'; await new Promise(r=>setTimeout(r,4000)); continue; }

    if (s.status === 'done') {
      st.textContent = '✅ Готово! (' + s.tiles_done + '/' + s.tiles_total + ' плиток)';
      dl.style.display = 'block';
      return;
    }
    if (s.status === 'error') {
      st.className = 'err';
      st.textContent = '❌ Помилка: ' + s.error;
      return;
    }
    st.textContent = '⏳ ' + s.stage + '  [' + s.tiles_done + '/' + s.tiles_total + ']';
    await new Promise(r => setTimeout(r, 4000));
  }
  st.textContent = 'Занадто довго — оновіть сторінку.';
}

// якщо задача вже йшла — одразу підхопити прогрес
poll();
</script></body></html>"""


@app.get("/precompute", response_class=HTMLResponse)
def precompute_page():
    return PRECOMPUTE_PAGE_HTML




# --------------------------------------------------------------------------
# Пошук ложбин для напівземлянок біля витоків яруг
# --------------------------------------------------------------------------
# Не весь загальний скан тераси, а ПРИЦІЛЬНИЙ пошук у безпосередній
# близькості (до ~150 м) від кожного вже знайденого витоку яру/джерела:
# невеличкий "кишеньковий" закуток, трохи вище джерела, захищений
# рельєфом від холодних вітрів (Пн/ПнСх/Сх) — типове місце напівземлянки.

HOLLOW_SEARCH_RADIUS_M = 150.0
HOLLOW_MAX_HEIGHT_ABOVE_SPRING_M = 6.0
HOLLOW_MIN_SCORE = 40.0
WIND_COLD_DIRS = [(1, 0), (1, 1), (0, 1)]  # grid-напрямки: r+1=Північ, c+1=Схід → Пн, ПнСх, Сх


def score_wind_shelter(grid: np.ndarray, r: int, c: int, cell_size_m: float, check_radius_m: float = 120.0) -> float:
    """
    Природний захист від холодних вітрів (Пн/ПнСх/Сх): чи є підвищення
    рельєфу в цих напрямках поблизу. Відкритість на Пд/ПдЗ (до долини/сонця)
    не карається — навпаки, це якраз бажано.
    """
    rows, cols = grid.shape
    z = grid[r, c]
    check_cells = max(1, int(check_radius_m / cell_size_m))
    protected = 0

    for dr, dc in WIND_COLD_DIRS:
        max_rise = 0.0
        for step in range(1, check_cells + 1):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < rows and 0 <= nc < cols:
                max_rise = max(max_rise, float(grid[nr, nc]) - float(z))
        if max_rise >= 2.5:  # відчутне підвищення — природний вітрозахист
            protected += 1

    return protected / len(WIND_COLD_DIRS)


def find_ravine_head_hollows(grid: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                              cell_size_m: float, head_indices: tuple):
    """Для кожного витоку яру шукає найкращу сусідню ложбину під напівземлянку."""
    rows, cols = grid.shape
    search_cells = max(2, int(HOLLOW_SEARCH_RADIUS_M / cell_size_m))
    hollows = []

    for hr, hc in head_indices:
        head_z = float(grid[hr, hc])
        best = None

        for dr in range(-search_cells, search_cells + 1):
            for dc in range(-search_cells, search_cells + 1):
                r, c = hr + dr, hc + dc
                if not (3 <= r < rows - 3 and 3 <= c < cols - 3):
                    continue
                dist_m = math.sqrt((dr * cell_size_m) ** 2 + (dc * cell_size_m) ** 2)
                if dist_m < 10.0 or dist_m > HOLLOW_SEARCH_RADIUS_M:
                    continue

                z = float(grid[r, c])
                delta_h = z - head_z
                if delta_h < 0.0 or delta_h > HOLLOW_MAX_HEIGHT_ABOVE_SPRING_M:
                    continue

                slope_deg, aspect_deg, _ = calc_geomorphology(grid, r, c, cell_size_m)
                if slope_deg > 10.0:
                    continue

                s_slope = score_slope_chernyakhiv(slope_deg)
                s_sun = score_sun_chernyakhiv(aspect_deg)
                s_wind = score_wind_shelter(grid, r, c, cell_size_m)
                s_close = max(0.0, 1.0 - dist_m / HOLLOW_SEARCH_RADIUS_M)
                s_height = max(0.0, 1.0 - delta_h / HOLLOW_MAX_HEIGHT_ABOVE_SPRING_M)

                hollow_score = (
                    0.30 * s_wind +
                    0.25 * s_close +
                    0.20 * s_height +
                    0.15 * s_slope +
                    0.10 * s_sun
                ) * 100

                if best is None or hollow_score > best["score"]:
                    best = {
                        "lat": round(float(lats[r]), 5),
                        "lon": round(float(lons[c]), 5),
                        "score": round(float(hollow_score), 1),
                        "dist_to_spring_m": round(dist_m),
                        "delta_h_above_spring_m": round(float(delta_h), 1),
                        "slope_deg": round(float(slope_deg), 1),
                        "aspect_deg": round(float(aspect_deg), 1),
                        "wind_shelter": round(float(s_wind), 2),
                        "spring_lat": round(float(lats[hr]), 5),
                        "spring_lon": round(float(lons[hc]), 5),
                    }

        if best is not None and best["score"] >= HOLLOW_MIN_SCORE:
            hollows.append(best)

    return hollows


@lru_cache(maxsize=32)
def get_hollows_for_area(lat: float, lon: float, radius_km: float):
    """Кешовано: ложбини для напівземлянок біля кожного витоку яру в зоні пошуку."""
    grid, lats, lons, cell_size_m = get_grid_bundle(lat, lon, radius_km)
    head_indices = get_channel_head_indices_for_scoring(lat, lon, radius_km, grid, lats, lons)
    hollows = find_ravine_head_hollows(grid, lats, lons, cell_size_m, head_indices)
    return tuple(hollows)


@lru_cache(maxsize=64)
def run_analysis(lat: float, lon: float, radius_km: float, culture: str = "kr") -> tuple:
    """Кешований аналіз. culture: 'kr' (Київська Русь), 'cherniakhiv' або 'both'."""
    grid, lats, lons, cell_size_m = get_grid_bundle(lat, lon, radius_km)

    # Для черняхівської культури спершу рахуємо яри/джерела (D8), а тоді
    # ще й придатні ложбини під напівземлянки біля цих джерел — і те, і те
    # використовується як фактор скорингу нижче.
    spring_dist_grid = None
    hollow_dist_grid = None
    if culture in ("cherniakhiv", "both"):
        head_indices = get_channel_head_indices_for_scoring(lat, lon, radius_km, grid, lats, lons)
        spring_dist_grid = build_distance_grid(grid, head_indices, cell_size_m)

        hollows = get_hollows_for_area(lat, lon, radius_km)
        hollow_indices = tuple(
            (int(np.argmin(np.abs(lats - h["lat"]))), int(np.argmin(np.abs(lons - h["lon"]))))
            for h in hollows
        )
        hollow_dist_grid = build_distance_grid(grid, hollow_indices, cell_size_m)

    rows, cols = grid.shape
    raw_kr, raw_chern = [], []

    for r in range(3, rows - 3):
        _report("scan", (r - 3) / max(1, rows - 6),
                f"Рядок {r - 2} з {rows - 6}, кандидатів: {len(raw_kr) + len(raw_chern)}")
        for c in range(3, cols - 3):
            lat_v = round(float(lats[r]), 5)
            lon_v = round(float(lons[c]), 5)

            if culture in ("kr", "both"):
                res = analyze_site_kr(grid, r, c, lat_v, lon_v, cell_size_m)
                if res and res["score"] >= MIN_SCORE:
                    raw_kr.append(res)

            if culture in ("cherniakhiv", "both"):
                dist_spring = float(spring_dist_grid[r, c]) if spring_dist_grid is not None else None
                dist_hollow = float(hollow_dist_grid[r, c]) if hollow_dist_grid is not None else None
                res_c = analyze_site_chernyakhiv(grid, r, c, lat_v, lon_v, cell_size_m, dist_spring, dist_hollow)
                if res_c and res_c["score"] >= MIN_SCORE:
                    raw_chern.append(res_c)

    limit = MAX_POINTS_BIG if radius_km >= 10 else MAX_POINTS_SMALL
    clean_kr = strict_nms_clustering(raw_kr, min_dist_m=NMS_KR_M)[:limit]
    clean_chern = strict_nms_clustering(raw_chern, min_dist_m=NMS_CHERN_M)[:limit]

    return tuple(clean_kr + clean_chern)


# --------------------------------------------------------------------------
# Ендпоінти
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Фонові розрахунки з прогрес-баром
# --------------------------------------------------------------------------
# /loading?<ті самі параметри, що й /map> — сторінка з прогрес-баром: запускає
# розрахунок у фоні (/api/job/start), опитує /api/job/status і, коли все готове
# (результати вже в lru_cache), перекидає на /map, яка відкривається миттєво.
# Solar2D може робити те саме: start -> опитувати status -> /api/analyze.

_jobs = {}
_jobs_lock = threading.Lock()
_job_sem = threading.Semaphore(1)  # одна важка задача за раз — на Render мало CPU/RAM


def _run_job(job_id, lat, lon, radius_km, culture, show_ravines, show_hollows, show_harvest):
    job = _jobs[job_id]

    def cb(phase, frac, detail):
        a, b, label = _PHASES[phase]
        job["percent"] = round((a + (b - a) * max(0.0, min(1.0, frac))) * 100, 1)
        job["stage"] = label
        job["detail"] = detail
        job["updated"] = time.time()

    job["stage"], job["detail"] = "У черзі", "Чекаю, поки завершиться попередній розрахунок"
    job["updated"] = time.time()
    with _job_sem:
        _tl.cb = cb
        try:
            cb("grid", 0.0, "Завантаження SRTM-тайлів (при першому запуску — довше)")
            grid, lats, lons, _cell = get_grid_bundle(lat, lon, radius_km)

            chern = culture in ("cherniakhiv", "both")
            if chern or show_ravines or show_hollows:
                cb("ravines", 0.0, "Аналіз стоку води…")
                get_channel_head_indices_for_scoring(lat, lon, radius_km, grid, lats, lons)
                if show_ravines:
                    get_ravines_for_area(lat, lon, radius_km)
                cb("ravines", 1.0, "")
            if chern or show_hollows:
                cb("hollows", 0.0, "Пошук ложбин біля джерел…")
                get_hollows_for_area(lat, lon, radius_km)
                cb("hollows", 1.0, "")

            cb("scan", 0.0, "Сканування рельєфу")
            results = run_analysis(lat, lon, radius_km, culture)
            job["count"] = len(results)

            if show_harvest:
                cb("harvest", 0.0, "Запит до Copernicus…")
                build_harvest_overlay(lat, lon, radius_km)
                cb("harvest", 1.0, "")

            job.update(status="done", percent=100.0, stage="Готово", detail=f"Знайдено точок: {len(results)}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            job.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            _tl.cb = None
            job["updated"] = time.time()


@app.get("/api/job/start")
def api_job_start(
    lat: float = Query(50.75, ge=-85, le=85),
    lon: float = Query(33.47, ge=-180, le=180),
    radius_km: float = Query(10.0, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM),
    culture: str = Query("kr", pattern="^(kr|cherniakhiv|both)$"),
    show_ravines: bool = Query(False),
    show_hollows: bool = Query(False),
    show_harvest: bool = Query(False),
):
    lat, lon, radius_km = round(lat, 5), round(lon, 5), round(radius_km, 2)
    key = (lat, lon, radius_km, culture, show_ravines, show_hollows, show_harvest)
    now = time.time()
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if now - v["updated"] > 3600]:
            del _jobs[jid]
        for jid, v in _jobs.items():
            if v["key"] == key and (v["status"] == "running" or (v["status"] == "done" and now - v["updated"] < 600)):
                return {"job_id": jid}
        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {"key": key, "status": "running", "percent": 0.0, "stage": "Запуск", "detail": "",
                         "t0": now, "updated": now, "count": None, "error": None}
    threading.Thread(target=_run_job, daemon=True,
                     args=(job_id, lat, lon, radius_km, culture, show_ravines, show_hollows, show_harvest)).start()
    return {"job_id": job_id}


@app.get("/api/job/status")
def api_job_status(id: str = Query(...)):
    job = _jobs.get(id)
    if job is None:
        return {"status": "unknown"}  # напр. сервер перезапустився — клієнт має стартувати заново
    now = time.time()
    return {"status": job["status"], "percent": job["percent"], "stage": job["stage"],
            "detail": job["detail"], "elapsed_s": round(now - job["t0"], 1),
            "idle_s": round(now - job["updated"], 1), "count": job["count"], "error": job["error"]}


@app.get("/loading", response_class=HTMLResponse)
def loading_page():
    return """
    <!DOCTYPE html>
    <html lang="uk"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GeoPredict — розрахунок</title>
    <style>
        body { font-family: sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; }
        .bar { height: 22px; border-radius: 11px; background: #8883; overflow: hidden; }
        .fill { height: 100%; width: 0; background: #2c7be5; transition: width .5s; }
        .pct { font-size: 32px; font-weight: bold; margin-top: 10px; }
        .muted { opacity: .7; font-size: 13px; margin-top: 6px; }
        .warn { color: #d9534f; opacity: 1; }
        .err { color: #d9534f; font-weight: bold; margin-top: 12px; }
    </style></head>
    <body>
        <h3>⏳ Аналіз рельєфу…</h3>
        <div class="bar"><div class="fill" id="fill"></div></div>
        <div class="pct" id="pct">0%</div>
        <div id="stage">Запуск…</div>
        <div class="muted" id="detail"></div>
        <div class="muted" id="timer"></div>
        <div class="muted warn" id="warn"></div>
        <div class="err" id="err"></div>
        <div class="muted">Перший запуск після простою може тривати до хвилини: Render «прокидає» сервер і качає дані SRTM.</div>
    <script>
        const $ = id => document.getElementById(id);
        const qs = location.search;
        const t0 = Date.now();
        let jobId = null, fails = 0;
        setInterval(() => { $('timer').textContent = 'Минуло: ' + Math.floor((Date.now() - t0) / 1000) + ' с'; }, 500);

        async function start() {
            try {
                const r = await fetch('/api/job/start' + qs);
                if (!r.ok) throw new Error('HTTP ' + r.status);
                jobId = (await r.json()).job_id;
                fails = 0; $('warn').textContent = '';
                poll();
            } catch (e) {
                fails++;
                $('warn').textContent = 'Сервер не відповідає (ймовірно, прокидається). Спроба ' + fails + '…';
                setTimeout(start, 3000);
            }
        }

        async function poll() {
            try {
                const r = await fetch('/api/job/status?id=' + jobId);
                const d = await r.json();
                if (d.status === 'unknown') { $('warn').textContent = 'Сервер перезапустився — стартую заново…'; return start(); }
                fails = 0;
                const p = Math.round(d.percent || 0);
                $('fill').style.width = p + '%';
                $('pct').textContent = p + '%';
                $('stage').textContent = d.stage || '';
                $('detail').textContent = d.detail || '';
                $('warn').textContent = d.idle_s > 20 ? 'Немає оновлень ' + Math.round(d.idle_s) + ' с — триває важкий крок (напр. завантаження SRTM), процес живий.' : '';
                if (d.status === 'error') { $('err').textContent = 'Помилка: ' + d.error; return; }
                if (d.status === 'done') { $('stage').textContent = 'Готово, відкриваю карту…'; location.replace('/map' + qs); return; }
                setTimeout(poll, 1000);
            } catch (e) {
                fails++;
                $('warn').textContent = 'Немає зв\u2019язку з сервером, спроба ' + fails + '…';
                setTimeout(poll, 3000);
            }
        }
        start();
    </script></body></html>
    """


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "GeoPredict API (КР + WMS NDVI) працює"}


_PLACE_RANK = {"city": 0, "town": 1, "village": 2, "hamlet": 3, "suburb": 4,
               "neighbourhood": 5, "isolated_dwelling": 6, "locality": 7}
_PLACE_UK = {"city": "місто", "town": "місто", "village": "село", "hamlet": "хутір",
             "suburb": "район", "neighbourhood": "мікрорайон",
             "isolated_dwelling": "двір/хутір", "locality": "урочище"}
SUMY_VIEWBOX = "32.4,52.2,35.8,49.9"          # Nominatim: lon1,lat1,lon2,lat2
SUMY_BBOX_OVERPASS = "49.9,32.4,52.2,35.8"    # Overpass: south,west,north,east
_OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                  "https://overpass.kumi.systems/api/interpreter"]
_PLACES_CACHE = {}


def _clean_place_query(q: str) -> str:
    # Раніше фронтенд додавав ", Сумська область" до запиту — через це Nominatim
    # майже не знаходив села. Прибираємо цей хвіст, рамки Сумщини задає viewbox.
    q = q.strip()
    q = re.sub(r"[,\s]*(сумська\s+обл(?:асть|\.)?|сумщина)\s*$", "", q, flags=re.IGNORECASE)
    return q.strip(" ,")


def _nominatim_places(q: str) -> list:
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": q, "format": "jsonv2", "countrycodes": "ua",
                "viewbox": SUMY_VIEWBOX, "bounded": 1, "limit": 20,
                "addressdetails": 1, "accept-language": "uk", "dedupe": 1},
        headers={"User-Agent": "GeoPredict/1.0 (archaeology research helper)"},
        timeout=12,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json():
        kind = next((k for k in (item.get("addresstype"), item.get("type")) if k in _PLACE_RANK), None)
        if kind is None:
            continue  # вулиці, магазини тощо — не населені пункти
        disp = item.get("display_name", "")
        name = item.get("name") or disp.split(",")[0]
        out.append({"name": name, "kind": kind, "display_name": disp,
                    "label": f"{_PLACE_UK.get(kind, kind)}: {disp}",
                    "lat": round(float(item["lat"]), 5), "lon": round(float(item["lon"]), 5)})
    return out


def _ci_regex(text: str) -> str:
    """Регістронезалежний regex через класи [Гг] — надійніше за ',i' для кирилиці в Overpass."""
    parts = []
    for ch in text:
        if ch.upper() != ch.lower():
            parts.append(f"[{ch.upper()}{ch.lower()}]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _overpass_places(q: str) -> list:
    """Overpass: назви, що ПОЧИНАЮТЬСЯ з q. Короткі таймаути — повільний Overpass не має гальмувати підказки."""
    safe = re.sub(r"[^0-9A-Za-zА-Яа-яІіЇїЄєҐґʼ'’\- ]", "", q)
    if len(safe) < 3:
        return []
    query = (
        '[out:json][timeout:7];'
        'nwr["place"~"^(city|town|village|hamlet|suburb|isolated_dwelling|locality)$"]'
        f'["name"~"^{_ci_regex(safe)}"]({SUMY_BBOX_OVERPASS});out center 40;'
    )
    elements = None
    for url in _OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, timeout=(3, 8),
                              headers={"User-Agent": "GeoPredict/1.0"})
            r.raise_for_status()
            elements = r.json().get("elements", [])
            break
        except Exception as exc:
            print(f"[places_search] Overpass {url}: {type(exc).__name__}")
    out = []
    for el in elements or []:
        tags = el.get("tags", {})
        kind = tags.get("place")
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if not kind or lat is None or lon is None or not tags.get("name"):
            continue
        name = tags["name"]
        out.append({"name": name, "kind": kind, "display_name": f"{name} ({lat:.4f}, {lon:.4f})",
                    "label": f"{_PLACE_UK.get(kind, kind)}: {name} ({lat:.3f}, {lon:.3f})",
                    "lat": round(float(lat), 5), "lon": round(float(lon), 5)})
    return out


def _photon_places(q: str) -> list:
    """Photon (komoot): пошук за ПОЧАТКОМ назви, тобто підходить для підказок під час набору."""
    resp = requests.get(
        "https://photon.komoot.io/api/",
        params={"q": q, "bbox": "32.4,49.9,35.8,52.2", "limit": 15, "osm_tag": "place"},
        headers={"User-Agent": "GeoPredict/1.0"}, timeout=(3, 7),
    )
    resp.raise_for_status()
    out = []
    for f in resp.json().get("features", []):
        p = f.get("properties", {})
        kind = p.get("osm_value")
        name = p.get("name")
        lon, lat = (f.get("geometry", {}).get("coordinates") or [None, None])[:2]
        if kind not in _PLACE_RANK or not name or lat is None:
            continue
        where = ", ".join(x for x in (p.get("district") or p.get("county"), p.get("state")) if x)
        out.append({"name": name, "kind": kind, "display_name": f"{name}, {where}" if where else name,
                    "label": f"{_PLACE_UK.get(kind, kind)}: {name}" + (f", {where}" if where else ""),
                    "lat": round(float(lat), 5), "lon": round(float(lon), 5)})
    return out


@app.get("/api/places/search")
def api_places_search(q: str = Query(..., min_length=2)):
    """
    Пошук населеного пункту (місто, село, хутір, урочище) у межах Сумщини:
    Nominatim + доповнення з Overpass (частковий збіг назви, дрібні села).
    """
    q_clean = _clean_place_query(q)
    if len(q_clean) < 2:
        return {"results": []}
    if q_clean.lower() in _PLACES_CACHE:
        return {"results": _PLACES_CACHE[q_clean.lower()]}

    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FTimeout
    found, errors = [], []
    pool = ThreadPoolExecutor(max_workers=3)
    futs = {pool.submit(fn, q_clean): name for name, fn in
            (("Nominatim", _nominatim_places), ("Photon", _photon_places), ("Overpass", _overpass_places))}
    try:
        for fut in as_completed(futs, timeout=6):
            try:
                found += fut.result()
            except Exception as exc:
                errors.append(f"{futs[fut]}: {type(exc).__name__}")
                print(f"[places_search] {futs[fut]}: {exc}")
    except FTimeout:
        print("[places_search] Дедлайн 6 с — віддаю те, що встигло.")
    pool.shutdown(wait=False)  # не чекаємо повільні джерела

    seen, results = set(), []
    ql = q_clean.lower()
    found.sort(key=lambda p: (not p["name"].lower().startswith(ql), _PLACE_RANK.get(p["kind"], 9), p["name"]))
    for p in found:
        key = (p["name"].lower(), round(p["lat"], 2), round(p["lon"], 2))
        if key in seen:
            continue
        seen.add(key)
        results.append({k: p[k] for k in ("name", "display_name", "label", "lat", "lon")})
        if len(results) >= 25:
            break

    if results:
        if len(_PLACES_CACHE) > 300:
            _PLACES_CACHE.clear()
        _PLACES_CACHE[ql] = results
    payload = {"results": results}
    if errors and not results:
        payload["detail"] = "Пошук тимчасово недоступний: " + "; ".join(errors)
    return payload


# --------------------------------------------------------------------------
# Історична довідка по точці (факти з Вікіпедії/OSM + стислий виклад через LLM)
# --------------------------------------------------------------------------
# Змінні середовища на Render:
#   LLM_API_KEY  (або GROQ_API_KEY) — ключ
#   LLM_BASE_URL — за замовчуванням Groq: https://api.groq.com/openai/v1
#   LLM_MODEL    — за замовчуванням openai/gpt-oss-120b (якщо Groq змінить назву — поміняйте тут)
# Модель отримує ЛИШЕ зібрані факти й не має права вигадувати знахідки та дати.

_INFO_CACHE = {}
_WIKI_API = "https://uk.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "GeoPredict/1.0 (archaeology research helper)"}


def _hav_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def _info_reverse(lat, lon) -> dict:
    r = requests.get("https://nominatim.openstreetmap.org/reverse",
                     params={"lat": lat, "lon": lon, "zoom": 14, "format": "jsonv2",
                             "accept-language": "uk", "addressdetails": 1},
                     headers=_WIKI_HEADERS, timeout=(3, 8))
    r.raise_for_status()
    a = r.json().get("address", {})
    place = a.get("village") or a.get("hamlet") or a.get("town") or a.get("city") or a.get("suburb")
    return {"place": place, "district": a.get("county") or a.get("municipality"), "state": a.get("state")}


def _info_wiki_geo(lat, lon) -> list:
    r = requests.get(_WIKI_API, params={"action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
                                        "gsradius": 8000, "gslimit": 12, "format": "json"},
                     headers=_WIKI_HEADERS, timeout=(3, 8))
    r.raise_for_status()
    return r.json().get("query", {}).get("geosearch", [])


def _info_wiki_text(pageid: int) -> str:
    """Розділ «Історія» зі статті (або початок статті)."""
    r = requests.get(_WIKI_API, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                        "pageids": pageid, "format": "json"},
                     headers=_WIKI_HEADERS, timeout=(3, 10))
    r.raise_for_status()
    page = next(iter(r.json().get("query", {}).get("pages", {}).values()), {})
    text = page.get("extract", "") or ""
    m = re.search(r"==\s*(Історія|Історичні відомості|Історичний нарис)[^=]*==\s*(.*?)(?:\n==[^=]|\Z)", text, re.S)
    if m and len(m.group(2).strip()) > 80:
        return m.group(2).strip()[:1800]
    return text.split("\n==")[0].strip()[:900]


def _info_overpass(lat, lon) -> list:
    q = (f'[out:json][timeout:6];nwr(around:2500,{lat},{lon})["historic"];out center 15;')
    r = requests.post(_OVERPASS_URLS[1], data={"data": q}, timeout=(3, 8), headers={"User-Agent": "GeoPredict/1.0"})
    r.raise_for_status()
    out = []
    for el in r.json().get("elements", []):
        t = el.get("tags", {})
        la = el.get("lat") or (el.get("center") or {}).get("lat")
        lo = el.get("lon") or (el.get("center") or {}).get("lon")
        if la is None or lo is None:
            continue
        out.append({"name": t.get("name", ""), "kind": t.get("historic", ""),
                    "dist_m": int(_hav_m(lat, lon, la, lo))})
    out.sort(key=lambda x: x["dist_m"])
    return out[:8]


def _find_llm_key() -> str:
    """Ключ шукаємо за кількома назвами змінних (регістр у Render має значення) і за префіксом gsk_ (Groq)."""
    for name in ("LLM_API_KEY", "GROQ_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    for name, val in os.environ.items():
        v = val.strip()
        if "GROQ" in name.upper() and v:
            return v
    for val in os.environ.values():
        v = val.strip()
        if v.startswith("gsk_") and len(v) > 30:
            return v
    return ""


def _llm_chat(system: str, user: str) -> str:
    key = _find_llm_key()
    if not key:
        raise RuntimeError("no_key")
    base = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    body = {"model": model, "temperature": 0.2, "max_tokens": 1500,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json=body, timeout=(5, 45))
    if r.status_code == 429:
        raise RuntimeError("rate_limit")
    if r.status_code >= 400:
        raise RuntimeError(f"http_{r.status_code}: {r.text[:200]}")
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


_INFO_SYSTEM = (
    "Ти — краєзнавець-помічник для археолога-аматора на Сумщині. Пиши українською, коротко (до 7 речень), без вступів. "
    "ПРАВИЛА: спирайся ТІЛЬКИ на надані факти. Не вигадуй знахідок, дат, назв поселень чи доріг, яких немає у фактах. "
    "Якщо про саме це місце нічого не відомо — прямо скажи це одним реченням. "
    "Можеш додати одне речення із загальним історичним контекстом регіону, але лише про те, що точно відомо, "
    "і познач його словами «Загальний контекст регіону:». "
    "Структура: 1) найближчі населені пункти і що відомо про їхню історію; 2) історичні об'єкти поруч (якщо є); "
    "3) що це означає для пошуку (за даними рельєфу). Не вживай форматування markdown."
)


@app.get("/api/point/info")
def api_point_info(
    lat: float = Query(..., ge=-85, le=85),
    lon: float = Query(..., ge=-180, le=180),
    kind: str = Query("", max_length=80),
    score: float = Query(0.0),
    delta_h: float = Query(0.0),
    dist_water: int = Query(0),
):
    lat, lon = round(lat, 5), round(lon, 5)
    ckey = (round(lat, 3), round(lon, 3), kind)
    if ckey in _INFO_CACHE:
        return _INFO_CACHE[ckey]

    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FTimeout
    warnings, rev, geo, hist = [], {}, [], []
    pool = ThreadPoolExecutor(max_workers=4)
    futs = {pool.submit(_info_reverse, lat, lon): "rev", pool.submit(_info_wiki_geo, lat, lon): "geo",
            pool.submit(_info_overpass, lat, lon): "hist"}
    try:
        for fut in as_completed(futs, timeout=9):
            try:
                res = fut.result()
                if futs[fut] == "rev": rev = res
                elif futs[fut] == "geo": geo = res
                else: hist = res
            except Exception as exc:
                print(f"[point_info] {futs[fut]}: {type(exc).__name__}: {exc}")
                if futs[fut] != "hist":  # OSM historic часто недоступний з Render — це не помилка для користувача
                    warnings.append(f"Джерело '{futs[fut]}' недоступне")
    except FTimeout:
        warnings.append("Частина джерел не відповіла вчасно")

    # тексти двох найближчих статей Вікіпедії (паралельно)
    geo = sorted(geo, key=lambda g: g.get("dist", 1e9))
    top = geo[:2]
    texts = {}
    tfuts = {pool.submit(_info_wiki_text, g["pageid"]): g["pageid"] for g in top}
    try:
        for fut in as_completed(tfuts, timeout=10):
            try:
                texts[tfuts[fut]] = fut.result()
            except Exception as exc:
                print(f"[point_info] wiki_text: {exc}")
    except FTimeout:
        pass
    pool.shutdown(wait=False)

    sources = [{"title": g["title"], "url": f"https://uk.wikipedia.org/?curid={g['pageid']}",
                "dist_m": int(g.get("dist", 0))} for g in geo[:6]]

    facts = [f"Координати точки: {lat}, {lon}.",
             f"Тип цілі за аналізом рельєфу: {kind or 'невідомо'}, бал {score}%, "
             f"перевищення над низиною {delta_h} м, до води ~{dist_water} м."]
    if rev.get("place"):
        facts.append(f"Найближчий населений пункт за OSM: {rev['place']} ({rev.get('district') or ''}, {rev.get('state') or ''}).")
    for g in geo[:6]:
        txt = texts.get(g["pageid"])
        line = f"Стаття Вікіпедії «{g['title']}», за {int(g.get('dist', 0))} м від точки."
        facts.append(line + (f" Текст: {txt}" if txt else ""))
    for h in hist:
        facts.append(f"OSM історичний об'єкт: {h['kind']} {h['name']} — за {h['dist_m']} м.")
    if not geo and not hist:
        facts.append("У радіусі 8 км статей Вікіпедії та історичних об'єктів OSM не знайдено.")

    summary, llm_error = None, None
    try:
        summary = _llm_chat(_INFO_SYSTEM, "ФАКТИ:\n" + "\n".join(facts) + "\n\nСкладіть довідку.")
    except Exception as exc:
        msg = str(exc)
        llm_error = ("На Render не знайдено ключ Groq (змінна GROQ_API_KEY)" if msg == "no_key" else
                     "Ліміт запитів до AI вичерпано, спробуйте за хвилину" if msg == "rate_limit" else
                     f"Помилка AI: {msg}")
        print(f"[point_info] LLM: {msg}")

    payload = {"summary": summary, "llm_error": llm_error, "place": rev.get("place"),
               "sources": sources, "historic_osm": hist, "warnings": warnings,
               "facts_found": len(geo) + len(hist)}
    if summary:  # кешуємо лише успішні відповіді
        if len(_INFO_CACHE) > 500:
            _INFO_CACHE.clear()
        _INFO_CACHE[ckey] = payload
    return payload


@app.get("/test", response_class=HTMLResponse)
def test_form():
    """
    Сторінка вибору місця: пошук населеного пункту замість ручного введення
    координат, плюс радіус, культура і опційні шари — і одразу готові
    посилання на /map та /api/analyze.
    """
    return """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>GeoPredict — вибір місця</title>
        <style>
            body { font-family: sans-serif; max-width: 480px; margin: 20px auto; padding: 0 16px; }
            label { display: block; margin-top: 14px; font-weight: bold; font-size: 14px; }
            input, select { width: 100%; padding: 10px; box-sizing: border-box; font-size: 16px; margin-top: 4px; }
            .suggestions { margin-top: 6px; }
            .suggestion-btn {
                display: block; width: 100%; text-align: left; padding: 10px;
                margin-top: 4px; border: 1px solid #8883; border-radius: 6px;
                background: transparent; font-size: 15px; cursor: pointer;
            }
            .suggestion-btn:active { background: #8882; }
            .picked { padding: 10px; border-radius: 6px; background: #2c7be522; margin-top: 8px; font-size: 14px; }
            .row { display: flex; gap: 10px; margin-top: 20px; }
            .row2 { display: flex; gap: 10px; margin-top: 10px; }
            .row2 > div { flex: 1; }
            a.btn {
                flex: 1; text-align: center; padding: 12px; border-radius: 6px;
                text-decoration: none; color: white; font-weight: bold;
            }
            .btn-map { background: #2c7be5; }
            .btn-json { background: #2f9e44; }
            .muted { opacity: .65; font-size: 13px; margin-top: 4px; }
        </style>
    </head>
    <body>
        <h2>📍 Оберіть місце пошуку</h2>

        <label for="place">Населений пункт (місто, село, хутір — Сумщина)</label>
        <input id="place" type="text" placeholder="напр. Ромни, Кролевець, будь-яке село...">
        <div class="suggestions" id="suggestions"></div>
        <div class="picked" id="picked" style="display:none;"></div>

        <div class="row2">
            <div>
                <label for="lat">Широта</label>
                <input id="lat" type="number" step="0.00001" value="50.75">
            </div>
            <div>
                <label for="lon">Довгота</label>
                <input id="lon" type="number" step="0.00001" value="33.47">
            </div>
        </div>
        <div class="muted">Можна й ввести вручну — поля вище завжди редаговані.</div>

        <label for="radius">Радіус пошуку (км)</label>
        <input id="radius" type="number" step="0.5" min="0.5" max="50" value="5">

        <label for="culture">Культура</label>
        <select id="culture">
            <option value="cherniakhiv">Черняхівська</option>
            <option value="kr">Київська Русь</option>
            <option value="both">Обидві</option>
        </select>

        <div class="row2">
            <div><label><input type="checkbox" id="ravines"> Яри/струмки</label></div>
            <div><label><input type="checkbox" id="hollows"> Ложбини (будиночки)</label></div>
        </div>
        <div class="row2">
            <div><label><input type="checkbox" id="harvest"> Скошені поля</label></div>
        </div>

        <div class="row">
            <a class="btn btn-map" id="mapLink" href="#" target="_blank">Відкрити карту</a>
            <a class="btn btn-json" id="jsonLink" href="#" target="_blank">Відкрити JSON</a>
        </div>

        <script>
            const $ = id => document.getElementById(id);
            let searchTimer = null;

            $('place').addEventListener('input', () => {
                clearTimeout(searchTimer);
                const q = $('place').value.trim();
                if (q.length < 2) { $('suggestions').innerHTML = ''; return; }
                searchTimer = setTimeout(() => runSearch(q), 700);
            });

            let searchCtl = null;
            async function runSearch(q) {
                if (searchCtl) searchCtl.abort();
                searchCtl = new AbortController();
                $('suggestions').innerHTML = '<div class="muted">Шукаю…</div>';
                try {
                    const r = await fetch('/api/places/search?q=' + encodeURIComponent(q), {signal: searchCtl.signal});
                    const data = await r.json();
                    const box = $('suggestions');
                    box.innerHTML = '';
                    if (!data.results || !data.results.length) {
                        box.innerHTML = '<div class="muted">Нічого не знайдено. Спробуйте іншу назву або введіть координати вручну.</div>';
                        return;
                    }
                    data.results.forEach(place => {
                        const b = document.createElement('button');
                        b.className = 'suggestion-btn';
                        b.type = 'button';
                        b.textContent = place.label || place.display_name;
                        b.onclick = () => pick(place);
                        box.appendChild(b);
                    });
                } catch (e) {
                    if (e.name === 'AbortError') return;
                    $('suggestions').innerHTML = '<div class="muted">Помилка пошуку. Введіть координати вручну.</div>';
                }
            }

            function pick(place) {
                $('lat').value = place.lat;
                $('lon').value = place.lon;
                $('suggestions').innerHTML = '';
                $('place').value = place.name;
                const p = $('picked');
                p.style.display = 'block';
                p.textContent = '✅ Обрано: ' + place.display_name + ' (' + place.lat + ', ' + place.lon + ')';
                updateLinks();
            }

            function updateLinks() {
                const q = `lat=${$('lat').value}&lon=${$('lon').value}&radius_km=${$('radius').value}&culture=${$('culture').value}`
                    + `&show_ravines=${$('ravines').checked}&show_hollows=${$('hollows').checked}&show_harvest=${$('harvest').checked}`;
                $('mapLink').href = `/loading?${q}`;
                $('jsonLink').href = `/api/analyze?lat=${$('lat').value}&lon=${$('lon').value}&radius_km=${$('radius').value}&culture=${$('culture').value}`;
            }

            ['lat','lon','radius','culture','ravines','hollows','harvest'].forEach(id =>
                $(id).addEventListener('input', updateLinks)
            );
            updateLinks();
        </script>
    </body>
    </html>
    """


@app.get("/api/inspect")
def api_inspect(
    points: str = Query(
        ...,
        description="Список точок через ';': lat1,lon1,мітка1;lat2,lon2,мітка2;...",
    ),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    """
    Службовий ендпоінт для калібрування порогів: показує СИРІ розраховані
    значення (висота, перепад, відстань до води, схил, азимут) для будь-якої
    точки — навіть якщо вона НЕ проходить поточні пороги в analyze_site_*.
    Третій (опційний) елемент кожної точки — довільна мітка (наприклад, назва
    з відомого каталогу пам'яток), просто повертається назад для звірки.
    format=csv — завантажує результат як CSV-файл (зручно на мобільному).
    """
    inspect_radius_km = 1.0
    grid_steps = choose_grid_steps(inspect_radius_km)

    out = []
    for raw_pt in points.split(";"):
        raw_pt = raw_pt.strip()
        if not raw_pt:
            continue
        parts = raw_pt.split(",")
        label = parts[2] if len(parts) >= 3 else ""
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            out.append({"input": raw_pt, "label": label, "error": "Очікується формат lat,lon[,мітка]"})
            continue

        try:
            grid, lats, lons, cell_size_m = get_grid_bundle(lat, lon, inspect_radius_km)
            r = int(np.argmin(np.abs(lats - lat)))
            c = int(np.argmin(np.abs(lons - lon)))

            z_center = float(grid[r, c])
            slope_deg, aspect_deg, _ = calc_geomorphology(grid, r, c, cell_size_m)

            search_r = max(3, min(12, int(450.0 / cell_size_m)))
            min_z, dist_water_m = nearest_lower_point(grid, r, c, search_r, cell_size_m)
            delta_h = z_center - min_z

            tip_score = detect_promontory_kr(grid, r, c)

            out.append({
                "label": label,
                "lat": lat,
                "lon": lon,
                "elevation_m": round(z_center, 1),
                "delta_h_m": round(delta_h, 1),
                "dist_water_m": round(dist_water_m),
                "slope_deg": round(slope_deg, 1),
                "aspect_deg": round(aspect_deg, 1),
                "kr_tip_score": round(float(tip_score), 2),
                "chern_s_height": round(score_height_chernyakhiv(delta_h), 2),
                "chern_s_water": round(score_water_chernyakhiv(dist_water_m), 2),
                "chern_s_slope": round(score_slope_chernyakhiv(slope_deg), 2),
                "chern_s_sun": round(score_sun_chernyakhiv(aspect_deg), 2),
            })
        except Exception as exc:
            out.append({"label": label, "lat": lat, "lon": lon, "error": str(exc)})

    if format == "csv":
        cols = ["label", "lat", "lon", "elevation_m", "delta_h_m", "dist_water_m",
                "slope_deg", "aspect_deg", "kr_tip_score", "chern_s_height",
                "chern_s_water", "chern_s_slope", "chern_s_sun", "error"]
        lines = [",".join(cols)]
        for row in out:
            lines.append(",".join(str(row.get(k, "")) for k in cols))
        csv_text = "\n".join(lines)
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=inspect_results.csv"},
        )

    return {"points": out}


@app.get("/api/hollows")
def api_hollows(
    lat: float = Query(50.75, ge=-85, le=85),
    lon: float = Query(33.47, ge=-180, le=180),
    radius_km: float = Query(5.0, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM),
):
    """
    Ложбини для напівземлянок біля витоків яруг — прицільний пошук у радіусі
    150м навколо кожного джерела: невисоко над водою, захищено від холодних
    вітрів (Пн/ПнСх/Сх), полога ділянка.
    """
    hollows = get_hollows_for_area(round(lat, 5), round(lon, 5), round(radius_km, 2))
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "count": len(hollows),
        "hollows": list(hollows),
    }


@app.get("/api/ravines")
def api_ravines(
    lat: float = Query(50.75, ge=-85, le=85),
    lon: float = Query(33.47, ge=-180, le=180),
    radius_km: float = Query(5.0, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM),
    download: bool = Query(
        False,
        description="Завантажити як JSON-файл — для одноразового прекомп'ютування "
                    "великого регіону й додавання в data/precomputed_ravines.json",
    ),
):
    """
    Реконструкція яруг/струмків методом D8 flow accumulation.
    'heads' — ймовірні початки яруг/джерела (де могли селитись черняхівці).
    'segments' — лінії тальвегів для візуалізації мережі стоку.
    Якщо область покрита прекомп'ютованою плиткою (data/precomputed_ravines.json) —
    відповідь миттєва, без живого SRTM/D8 обчислення.
    """
    heads, segments = get_ravines_for_area(round(lat, 5), round(lon, 5), round(radius_km, 2))
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon, radius_km)

    payload = {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "bounds": {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon},
        "heads_count": len(heads),
        "heads": list(heads),
        "segments": [list(s) for s in segments],
    }

    if download:
        # Формат "tiles": [...] — саме такий, який очікує data/precomputed_ravines.json.
        tile_payload = {"tiles": [{
            "center": payload["center"],
            "radius_km": payload["radius_km"],
            "bounds": payload["bounds"],
            "heads": payload["heads"],
            "segments": payload["segments"],
        }]}
        return Response(
            content=json.dumps(tile_payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=precomputed_ravines_tile.json"},
        )

    return payload


@app.get("/api/analyze")
def api_analyze(
    lat: float = Query(50.75, ge=-85, le=85),
    lon: float = Query(33.47, ge=-180, le=180),
    radius_km: float = Query(10.0, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM),
    culture: str = Query("kr", pattern="^(kr|cherniakhiv|both)$"),
):
    """
    Легкий JSON-ендпоінт для мобільного клієнта (Solar2D).
    culture: 'kr' — Київська Русь, 'cherniakhiv' — черняхівська культура, 'both' — обидві.
    """
    results = run_analysis(round(lat, 5), round(lon, 5), round(radius_km, 2), culture)
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "culture": culture,
        "count": len(results),
        "results": list(results),
    }


@app.api_route("/map", methods=["GET", "HEAD"], response_class=HTMLResponse)
def get_map(
    lat: float = Query(50.75, ge=-85, le=85),
    lon: float = Query(33.47, ge=-180, le=180),
    radius_km: float = Query(10.0, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM),
    culture: str = Query("kr", pattern="^(kr|cherniakhiv|both)$"),
    show_ravines: bool = Query(False, description="Яри/струмки (D8) — повільно, вимкнено за замовчуванням"),
    show_hollows: bool = Query(False, description="Ложбини для напівземлянок біля витоків яруг"),
    show_harvest: bool = Query(False, description="Шар 'скошені поля' (4 запити до Copernicus) — повільно"),
):
    """HTML-мапа лишається для власного дебагу в браузері — Solar2D її не використовує."""
    results = run_analysis(round(lat, 5), round(lon, 5), round(radius_km, 2), culture)

    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    bounds = [[lat - lat_delta, lon - lon_delta], [lat + lat_delta, lon + lon_delta]]

    m = folium.Map(
        location=[lat, lon],
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🌍 Супутник HD (Esri)",
    )
    # Одразу вписуємо вид карти РІВНО в квадрат пошуку — інакше Leaflet
    # підвантажує NDVI/супутникові тайли з набагато ширшої області, ніж
    # реально потрібно, і це зайве навантаження на телефон і на Copernicus.
    m.fit_bounds(bounds)
    m.options["maxBounds"] = bounds
    m.options["maxBoundsViscosity"] = 0.6  # м'яко "пружинить" назад, а не жорстко блокує

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

    if COPERNICUS_INSTANCE_ID:
        sh_url = f"https://sh.dataspace.copernicus.eu/ogc/wms/{COPERNICUS_INSTANCE_ID}"
        time_range = sentinel_time_range()

        # Чіткий NDVI Sentinel-2 (10 м/піксель), автоматично бере найсвіжіший
        # безхмарний знімок з вікна CLOUD_SEARCH_WINDOW_DAYS.
        WmsTileLayer(
            url=sh_url,
            layers="VEGETATION_INDEX",
            name="🌱 NDVI Sentinel-2 (Copernicus, 10м)",
            fmt="image/png",
            transparent=True,
            overlay=True,
            opacity=0.75,
            attr="Copernicus Sentinel Data / Sentinel Hub",
            TIME=time_range,
            PRIORITY="mostRecent",
            MAXCC=MAX_CLOUD_PCT,
        ).add_to(m)

    else:
        # Запасний варіант без реєстрації — грубіший (250-500 м/піксель).
        WmsTileLayer(
            url="https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
            layers="MODIS_Terra_NDVI_8Day",
            name="🌱 NDVI MODIS (запасний, 250-500м)",
            fmt="image/png",
            transparent=True,
            overlay=True,
            opacity=0.75,
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

    center_research_url = f"{PUBLIC_BASE_URL}/research?lat={lat}&lon={lon}&radius_km={min(radius_km, 15.0)}"
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"Центр аналізу ({lat:.5f}, {lon:.5f})<br>"
            f"<a href='{center_research_url}' target='_blank'>🔎 Що писали про цю зону (AI)</a>",
            max_width=260,
        ),
        icon=folium.Icon(color="black", icon="info-sign"),
    ).add_to(m)

    for idx, pt in enumerate(results, 1):
        is_chernyakhiv = pt.get("culture") == "cherniakhiv"

        if is_chernyakhiv:
            # Фіолетова гама — черняхівська культура (відкриті поселення на терасах)
            color = "purple" if pt["score"] >= 68 else "mediumpurple" if pt["score"] >= 52 else "plum"
            type_label = "Черняхівське поселення (тераса)"
        else:
            # Червоно-синя гама — Київська Русь (оборонні миси)
            if pt["score"] >= 68:
                color = "red"
            elif pt["score"] >= 52:
                color = "orange"
            else:
                color = "darkblue"
            type_label = "Оборонний мис (Городище)" if pt.get("is_tip") else "Терасове селище"

        spring_line = ""
        if pt.get("dist_spring_m") is not None:
            spring_line = f"<b>До витоку яру/джерела:</b> ~{pt['dist_spring_m']} м<br>"

        hollow_line = ""
        if pt.get("dist_hollow_m") is not None:
            hollow_line = f"<b>До ложбини під напівземлянку:</b> ~{pt['dist_hollow_m']} м<br>"

        research_url = f"{PUBLIC_BASE_URL}/research?lat={pt['lat']}&lon={pt['lon']}&radius_km=3"

        popup_html = f"""
        <div style='font-family: sans-serif; width: 260px;'>
            <h4 style='margin:0 0 5px 0; color:#d9534f;'>Ціль #{idx} (Бал: {pt['score']}%)</h4>
            <b>Тип:</b> {type_label}<br>
            <b>Висота над низиною:</b> +{pt['delta_h_m']} м<br>
            <b>Абс. висота:</b> {pt['elevation_m']} м<br>
            <b>До річки/заплави:</b> ~{pt['dist_water_m']} м<br>
            {spring_line}
            {hollow_line}
            <b>Схил / Сонце:</b> {pt['aspect_deg']}° ({int(pt['sun_score'] * 100)}%)<br>
            <button onclick="geoInfo(this)" data-lat="{pt['lat']}" data-lon="{pt['lon']}"
                data-kind="{type_label}" data-score="{pt['score']}" data-dh="{pt['delta_h_m']}" data-dw="{pt['dist_water_m']}"
                style="margin-top:6px;padding:6px 10px;border:0;border-radius:6px;background:#2c7be5;color:#fff;font-size:13px;">
                📜 Історична довідка</button>
            <div class="geo-info" style="margin-top:6px;max-height:230px;overflow:auto;font-size:12px;line-height:1.35;"></div>
        </div>
        """

        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=7 if radius_km > 10 else 9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    if show_harvest:
        harvest_overlay = build_harvest_overlay(round(lat, 5), round(lon, 5), round(radius_km, 2))
        if harvest_overlay is not None:
            ImageOverlay(
                image=harvest_overlay,
                bounds=bounds,
                opacity=1.0,
                name="🌾 Скошені/зібрані ділянки (падіння NDVI від піку)",
            ).add_to(m)

    if show_ravines:
        ravine_heads, ravine_segments = get_ravines_for_area(round(lat, 5), round(lon, 5), round(radius_km, 2))
        ravine_fg = folium.FeatureGroup(name="🏞️ Яри/струмки (реконструкція стоку)")
        for p1, p2 in ravine_segments:
            folium.PolyLine([p1, p2], color="#1f78ff", weight=2, opacity=0.55).add_to(ravine_fg)
        for h in ravine_heads:
            folium.CircleMarker(
                location=[h["lat"], h["lon"]],
                radius=5,
                color="#00bcd4",
                fill=True,
                fill_color="#00bcd4",
                fill_opacity=0.95,
                popup=f"Ймовірний початок яру / джерело<br>Висота: {h['elevation_m']} м",
                tooltip="Джерело / початок яру",
            ).add_to(ravine_fg)
        ravine_fg.add_to(m)

    if show_hollows:
        hollows = get_hollows_for_area(round(lat, 5), round(lon, 5), round(radius_km, 2))
        hollow_fg = folium.FeatureGroup(name="🏚️ Ложбини для напівземлянок")
        for h in hollows:
            popup_html = f"""
            <div style='font-family: sans-serif; width: 220px;'>
                <h4 style='margin:0 0 5px 0; color:#8d6e63;'>Ложбина (Бал: {h['score']}%)</h4>
                <b>До джерела:</b> {h['dist_to_spring_m']} м<br>
                <b>Вище джерела на:</b> {h['delta_h_above_spring_m']} м<br>
                <b>Схил:</b> {h['slope_deg']}°<br>
                <b>Захист від вітру:</b> {int(h['wind_shelter'] * 100)}%
            </div>
            """
            folium.Marker(
                location=[h["lat"], h["lon"]],
                icon=folium.Icon(color="beige", icon="home", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Ложбина під напівземлянку (Бал: {h['score']}%)",
            ).add_to(hollow_fg)
        hollow_fg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    html = m.get_root().render()

    mobile_fix = (
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "<style>"
        "html,body{margin:0;padding:0;width:100%;height:100%;} "
        ".folium-map{width:100% !important;height:100% !important;} "
        # Список шарів раніше вилазив за межі екрана на вузьких телефонах —
        # обмежуємо ширину видимою областю й даємо горизонтальний скрол як запобіжник.
        ".leaflet-control-layers-expanded{"
        "max-width:82vw !important;max-height:70vh !important;overflow:auto !important;"
        "font-size:13px !important;box-sizing:border-box !important;white-space:normal !important;}"
        ".leaflet-top.leaflet-right{right:4px !important;left:auto !important;max-width:82vw;}"
        "</style>\n"
    )
    html = html.replace("<head>", "<head>\n" + mobile_fix, 1)

    # Маркери ложбин/яруг інколи не показувались одразу після завантаження —
    # класична проблема Leaflet: контейнер карти обчислює розмір ДО того, як
    # застосувався CSS на весь екран, і частина маркерів опиняється поза
    # видимою (на той момент коротшою) областю. invalidateSize() після
    # повного завантаження сторінки примушує Leaflet перерахувати розміри.
    m_match = re.search(r"var (map_[0-9a-f]+) = L\.map", html)
    if m_match:
        map_var = m_match.group(1)
        fix_script = (
            f"<script>window.addEventListener('load', function() {{ "
            f"setTimeout(function() {{ {map_var}.invalidateSize(); }}, 250); "
            f"}});</script>\n"
        )
        html = html.replace("</body>", fix_script + "</body>", 1)
        info_script = (
            "<script>var __geoMap = " + map_var + ";\n"
            "function geoEsc(s){return String(s).replace(/[&<>\"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c];});}\n"
            "function geoInfo(btn){\n"
            "  var box=btn.parentElement.querySelector('.geo-info'), d=btn.dataset, t0=Date.now();\n"
            "  btn.disabled=true; btn.style.opacity=0.6;\n"
            "  var tm=setInterval(function(){box.textContent='⏳ Збираю відомості… '+Math.round((Date.now()-t0)/1000)+' с';},500);\n"
            "  box.textContent='⏳ Збираю відомості…';\n"
            "  var q='lat='+d.lat+'&lon='+d.lon+'&kind='+encodeURIComponent(d.kind)+'&score='+d.score+'&delta_h='+d.dh+'&dist_water='+d.dw;\n"
            "  fetch('" + PUBLIC_BASE_URL + "/api/point/info?'+q).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json();})\n"
            "  .then(function(j){\n"
            "    var h='';\n"
            "    if(j.summary) h+='<div>'+geoEsc(j.summary).replace(/\\n/g,'<br>')+'</div>';\n"
            "    else h+='<div style=\"color:#d9534f\">'+geoEsc(j.llm_error||'AI не відповів')+'</div>';\n"
            "    if(j.sources&&j.sources.length){h+='<div style=\"margin-top:6px\"><b>Джерела (Вікіпедія):</b><br>';\n"
            "      j.sources.forEach(function(s){h+='<a href=\"'+s.url+'\" target=\"_blank\">'+geoEsc(s.title)+'</a> <span style=\"opacity:.6\">'+s.dist_m+' м</span><br>';});h+='</div>';}\n"
            "    h+='<div style=\"margin-top:6px;opacity:.6\">AI-довідка зі знайдених джерел. Це не доказ наявності пам\u2019ятки — перевіряйте.</div>';\n"
            "    box.innerHTML=h; btn.textContent='🔄 Оновити'; btn.disabled=false; btn.style.opacity=1;\n"
            "  }).catch(function(e){box.textContent='Помилка: '+e.message+'. Сервер, можливо, прокидається — спробуйте ще раз.';btn.disabled=false;btn.style.opacity=1;})\n"
            "  .finally(function(){clearInterval(tm); try{if(__geoMap._popup)__geoMap._popup.update();}catch(e){}});\n"
            "}</script>\n"
        )
        html = html.replace("</body>", info_script + "</body>", 1)

    return HTMLResponse(content=html)
