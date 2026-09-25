from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import srtm
import math
import folium
import requests
from datetime import datetime, timedelta

app = FastAPI()
elevation_data = srtm.get_data()

def get_sentinel2_region_metadata(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    """
    1 глобальний запит STAC для всього BBOX регіону (до 50 км).
    Отримує дані найновішого супутникового знімка Sentinel-2.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        
        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "datetime": f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
            "query": {"eo:cloud_cover": {"lt": 15}},
            "sortby": [{"field": "datetime", "direction": "desc"}],
            "limit": 1
        }
        
        resp = requests.post(stac_url, json=payload, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            if features:
                feat = features[0]
                props = feat.get("properties", {})
                scene_date = props.get("datetime", "")[:10]
                cloud_cover = round(props.get("eo:cloud_cover", 0), 1)
                
                return {
                    "date": scene_date,
                    "cloud": cloud_cover,
                    "tile_id": feat.get("id", "Sentinel-2"),
                    "status": "Оригінальний Sentinel-2 L2A"
                }
    except Exception:
        pass
        
    return {
        "date": "Актуальний сезон",
        "cloud": 0.0,
        "tile_id": "N/A",
        "status": "Базовий супутниковий фон"
    }

def calc_geomorphology(grid, r, c, cell_size_m):
    dz_dx = (grid[r][c+1] - grid[r][c-1]) / (2 * cell_size_m)
    dz_dy = (grid[r+1][c] - grid[r-1][c]) / (2 * cell_size_m)
    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_deg = math.degrees(math.atan2(-dz_dy, dz_dx)) % 360
    
    aspect_rad = math.radians(aspect_deg)
    ideal_rad = math.radians(135.0)  # Пд-Сх
    sun_score = max(0.0, math.cos(aspect_rad - ideal_rad))
    return slope_deg, aspect_deg, round(sun_score, 2)

def detect_promontory_tip(grid, r, c):
    z = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    dirs = [(-2,0), (2,0), (0,-2), (0,2), (-2,-2), (-2,2), (2,-2), (2,2)]
    lower_sectors = 0
    
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if z - grid[nr][nc] >= 7.0:
                lower_sectors += 1
                
    if 4 <= lower_sectors <= 6:
        return 1.0  # Ніс мису
    elif lower_sectors == 3:
        return 0.5  # Кромка
    else:
        return 0.0

def analyze_site(grid, r, c, lat_v, lon_v, cell_size_m):
    z_center = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    
    slope_deg, aspect_deg, sun_score = calc_geomorphology(grid, r, c, cell_size_m)
    tip_score = detect_promontory_tip(grid, r, c)
    
    if tip_score == 0.0:
        return None
        
    min_z = z_center
    dist_to_water_m = 999.0
    
    # Адаптивний радіус пошуку низини
    search_r = max(4, int(300.0 / cell_size_m))
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
    
    s_height = 1.0 if 10.0 <= delta_h <= 30.0 else (0.4 if 6.0 <= delta_h < 10.0 or 30.0 < delta_h <= 45.0 else 0.0)
    s_water = 1.0 if 50.0 <= dist_to_water_m <= 250.0 else (0.4 if 250.0 < dist_to_water_m <= 400.0 else 0.0)
    
    if s_height == 0.0 or s_water == 0.0:
        return None
        
    base_score = (0.40 * tip_score + 0.25 * s_water + 0.20 * s_height + 0.15 * sun_score) * 100
    
    return {
        "lat": lat_v,
        "lon": lon_v,
        "score": round(base_score, 1),
        "elevation_m": z_center,
        "delta_h_m": round(delta_h, 1),
        "dist_water_m": round(dist_to_water_m),
        "slope_deg": round(slope_deg, 1),
        "aspect_deg": round(aspect_deg, 1),
        "sun_score": sun_score,
        "is_tip": True if tip_score == 1.0 else False
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
    # Адаптивний крок сітки: для 1.5 км = 30 м, для 30 км = 140 м
    cell_size_m = max(30.0, radius_km * 4.8)
    
    lat_step = cell_size_m / 111000.0
    lon_step = cell_size_m / (111000.0 * math.cos(math.radians(lat)))
    
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    min_lat, max_lat = lat - lat_delta, lat + lat_delta
    min_lon, max_lon = lon - lon_delta, lon + lon_delta
    
    # 1. Запит супутникових метаданих 1 раз на весь BBOX
    sentinel_info = get_sentinel2_region_metadata(min_lat, min_lon, max_lat, max_lon)
    
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
            if res and res["score"] >= 68.0:
                res["sentinel_date"] = sentinel_info["date"]
                res["sentinel_cloud"] = sentinel_info["cloud"]
                raw_results.append(res)
                
    # Адаптивна дистанція придушення залежно від масштабу
    nms_dist = max(250.0, radius_km * 18.0)
    clean_results = strict_nms_clustering(raw_results, min_dist_m=nms_dist)
    
    limit = 15 if radius_km > 10 else 7
    return clean_results[:limit]

@app.get("/")
def read_root():
    return {"status": "GeoPredict API 30km Ready"}

@app.get("/map", response_class=HTMLResponse)
def get_map(lat: float, lon: float, radius_km: float = 30.0):
    results = run_analysis(lat, lon, radius_km)
    
    m = folium.Map(location=[lat, lon], zoom_start=10 if radius_km > 10 else 13, tiles=None)
    
    # Додаємо шари для ноутбука
    folium.TileLayer("OpenStreetMap", name="Топо-карта (OSM)").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Супутник HD (Esri)"
    ).add_to(m)
    
    folium.Marker(
        [lat, lon],
        popup=f"Центр аналізу (Радіус {radius_km} км)",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)
    
    for idx, pt in enumerate(results, 1):
        color = "red" if pt["score"] >= 82 else "orange"
        
        popup_html = f"""
        <div style='font-family: sans-serif; width: 220px;'>
            <h4 style='margin:0 0 5px 0; color:#d9534f;'>Ціль #{idx} (Бал: {pt['score']}%)</h4>
            <b>Тип:</b> {'Ніс мису (Край)' if pt['is_tip'] else 'Кромка тераси'}<br>
            <b>Sentinel-2 дата:</b> {pt.get('sentinel_date', 'N/A')}<br>
            <b>Хмарність:</b> {pt.get('sentinel_cloud', 0)}%<br>
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
        
    folium.LayerControl().add_to(m)
    return m._repr_html_()
