from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import srtm
import math
import folium
import requests
from datetime import datetime, timedelta

app = FastAPI()

# Кешування даних висот SRTM
elevation_data = srtm.get_data()

def get_sentinel2_ndvi_layer(lat: float, lon: float, radius_km: float):
    """
    Легкий запит до Microsoft Planetary Computer через requests (без важких бібліотек).
    Генерує URL шару NDVI для скошених полів / стерні / оранки (0.10 - 0.35).
    """
    try:
        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        bbox = [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]
        
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
            "query": {"eo:cloud_cover": {"lt": 20}},
            "sortby": [{"field": "datetime", "direction": "desc"}],
            "limit": 1
        }
        
        resp = requests.post(stac_url, json=payload, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            if features:
                item_id = features[0]["id"]
                scene_date = features[0]["properties"].get("datetime", "")[:10]
                
                # Dynamic Titiler URL під діапазон NDVI 0.10 - 0.35 (скошені поля)
                tile_url = (
                    f"https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x"
                    f"?collection=sentinel-2-l2a&item={item_id}"
                    f"&expression=(B08-B04)/(B08+B04)"
                    f"&rescale=0.10,0.35"
                    f"&colormap_name=YlOrRd"
                )
                return tile_url, scene_date
    except Exception:
        pass
        
    return None, None

def calc_geomorphology(grid, r, c, cell_size_m):
    dz_dx = (grid[r][c+1] - grid[r][c-1]) / (2 * cell_size_m)
    dz_dy = (grid[r+1][c] - grid[r-1][c]) / (2 * cell_size_m)
    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_deg = math.degrees(math.atan2(-dz_dy, dz_dx)) % 360
    
    aspect_rad = math.radians(aspect_deg)
    ideal_rad = math.radians(135.0)  # Південно-східне сонце
    sun_score = max(0.0, math.cos(aspect_rad - ideal_rad))
    return slope_deg, aspect_deg, round(sun_score, 2)

def detect_promontory(grid, r, c):
    """ Гнучкий детектор виступів, носів мисів та кромок ярів """
    z = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    lower_count = 0
    
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if z - grid[nr][nc] >= 3.5:  # Перепад від 3.5 метрів
                lower_count += 1
                
    if lower_count >= 3:
        return min(1.0, lower_count / 5.0)
    return 0.0

def analyze_site(grid, r, c, lat_v, lon_v, cell_size_m):
    z_center = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    
    tip_score = detect_promontory(grid, r, c)
    if tip_score == 0.0:
        return None
        
    slope_deg, aspect_deg, sun_score = calc_geomorphology(grid, r, c, cell_size_m)
    
    # Пошук низини/водотоку в радіусі до 350 м
    min_z = z_center
    dist_to_water_m = 999.0
    search_r = max(3, int(350.0 / cell_size_m))
    
    for dr in range(-search_r, search_r + 1):
        for dc in range(-search_r, search_r + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                val = grid[nr][nc]
                dist_m = math.sqrt((dr * cell_size_m)**2 + (dc * cell_size_m)**2)
                if val < min_z:
                    min_z = val
                    dist_to_water_m = dist_m
                    
    delta_h = z_center - min_z
    if delta_h < 4.0: # Мінімальний командний підйом над низиною
        return None
        
    # Плавні оцінки висоти та відстані до води (без жорстких відсікань)
    s_height = min(1.0, delta_h / 18.0) if delta_h <= 25.0 else max(0.3, 1.0 - (delta_h - 25.0) / 30.0)
    s_water = max(0.2, 1.0 - abs(dist_to_water_m - 150.0) / 300.0)
    
    final_score = (0.40 * tip_score + 0.25 * s_water + 0.20 * s_height + 0.15 * sun_score) * 100
    
    return {
        "lat": lat_v,
        "lon": lon_v,
        "score": round(final_score, 1),
        "elevation_m": z_center,
        "delta_h_m": round(delta_h, 1),
        "dist_water_m": round(dist_to_water_m),
        "slope_deg": round(slope_deg, 1),
        "aspect_deg": round(aspect_deg, 1),
        "sun_score": sun_score,
        "is_tip": True if tip_score >= 0.8 else False
    }

def strict_nms_clustering(results, min_dist_m):
    results.sort(key=lambda x: x["score"], reverse=True)
    filtered = []
    for pt in results:
        keep = True
        for existing in filtered:
            d_lat = (pt["lat"] - existing["lat"]) * 111000
            d_lon = (pt["lon"] - existing["lon"]) * 111000 * math.cos(math.radians(pt["lat"]))
            dist = math.sqrt(d_lat**2 + d_lon**2)
            if dist < min_dist_m:
                keep = False
                break
        if keep:
            filtered.append(pt)
    return filtered

def run_analysis(lat: float, lon: float, radius_km: float):
    # Оптимальний крок сітки для точності рельєфу (не більше 50-60 м)
    cell_size_m = 35.0 if radius_km <= 3.0 else (50.0 if radius_km <= 15.0 else 65.0)
    
    lat_step = cell_size_m / 111000.0
    lon_step = cell_size_m / (111000.0 * math.cos(math.radians(lat)))
    
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    min_lat, max_lat = lat - lat_delta, lat + lat_delta
    min_lon, max_lon = lon - lon_delta, lon + lon_delta
    
    grid, lats, lons = [], [], []
    curr_lat = min_lat
    
    while curr_lat <= max_lat:
        lats.append(curr_lat)
        row = []
        curr_lon = min_lon
        while curr_lon <= max_lon:
            if len(lons) < len(row) + 1:
                lons.append(curr_lon)
            alt = elevation_data.get_elevation(curr_lat, curr_lon)
            row.append(alt if alt is not None else 0)
            curr_lon += lon_step
        grid.append(row)
        curr_lat += lat_step
        
    rows, cols = len(grid), len(grid[0]) if grid else 0
    raw_results = []
    
    for r in range(2, rows - 2):
        for c in range(2, cols - 2):
            lat_v = round(lats[r], 5)
            lon_v = round(lons[c], 5)
            res = analyze_site(grid, r, c, lat_v, lon_v, cell_size_m)
            if res and res["score"] >= 50.0:  # Стабільний поріг для відбору
                raw_results.append(res)
                
    nms_dist = max(200.0, radius_km * 12.0)
    clean_results = strict_nms_clustering(raw_results, min_dist_m=nms_dist)
    
    limit = 15 if radius_km > 10 else 8
    return clean_results[:limit]

@app.get("/")
def read_root():
    return {"status": "GeoPredict API працює стабільно"}

@app.get("/map", response_class=HTMLResponse)
def get_map(lat: float = 50.75, lon: float = 33.47, radius_km: float = 10.0):
    # 1. Пошук кращих перспективних точок (мисів)
    results = run_analysis(lat, lon, radius_km)
    
    # 2. Шар NDVI для скошених полів
    ndvi_tile_url, scene_date = get_sentinel2_ndvi_layer(lat, lon, radius_km)
    
    # 3. Створення карти
    m = folium.Map(location=[lat, lon], zoom_start=11 if radius_km > 10 else 13, tiles=None)
    
    folium.TileLayer("OpenStreetMap", name="Топо-карта (OSM)").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Супутник HD (Esri)"
    ).add_to(m)
    
    if ndvi_tile_url:
        folium.TileLayer(
            tiles=ndvi_tile_url,
            attr="Copernicus Sentinel-2",
            name=f"🌾 Скошені поля/Оранка (Sentinel-2, {scene_date})",
            overlay=True,
            opacity=0.65
        ).add_to(m)
    
    folium.Marker(
        [lat, lon],
        popup=f"Центр аналізу (Радіус {radius_km} км)",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)
    
    for idx, pt in enumerate(results, 1):
        color = "red" if pt["score"] >= 75 else "orange"
        
        popup_html = f"""
        <div style='font-family: sans-serif; width: 220px;'>
            <h4 style='margin:0 0 5px 0; color:#d9534f;'>Ціль #{idx} (Бал: {pt['score']}%)</h4>
            <b>Тип:</b> {'Ніс мису (Край)' if pt['is_tip'] else 'Кромка тераси'}<br>
            <b>Сонце:</b> {pt['aspect_deg']}° (Світло: {int(pt['sun_score']*100)}%)<br>
            <b>Висота:</b> {pt['elevation_m']} м (+{pt['delta_h_m']} м)<br>
            <b>До води:</b> ~{pt['dist_water_m']} м
        </div>
        """
        
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=8 if radius_km > 10 else 10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)
        
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()
