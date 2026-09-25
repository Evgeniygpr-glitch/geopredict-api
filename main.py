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

def get_copernicus_s2_ndvi_layer(lat: float, lon: float, radius_km: float):
    """
    Прямий запит до Copernicus Data Space Ecosystem (CDSE) STAC API
    для отримання актуальних знімків Sentinel-2 L2A.
    """
    try:
        stac_url = "https://stac.dataspace.copernicus.eu/v1/search"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        bbox = [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]
        
        payload = {
            "collections": ["SENTINEL-2"],
            "bbox": bbox,
            "datetime": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
            "filter-lang": "cql2-text",
            "filter": "eo:cloud_cover < 25",
            "limit": 1
        }
        
        headers = {"Content-Type": "application/json"}
        resp = requests.post(stac_url, json=payload, headers=headers, timeout=3.5)
        
        scene_date = "Copernicus"
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            if features:
                scene_date = features[0]["properties"].get("datetime", "")[:10]

        # WMS шар Copernicus Data Space Ecosystem для NDVI / Sentinel-2
        copernicus_wms_url = (
            "https://sh.dataspace.copernicus.eu/ogc/wms/109312b9-2c0d-408a-a4e8-8b96e479c83f"
            "?SERVICE=WMS&REQUEST=GetMap&LAYERS=NDVI"
            "&MAXCC=20&WIDTH=512&HEIGHT=512&FORMAT=image/png"
            "&TIME=" + start_date.strftime("%Y-%m-%d") + "/" + end_date.strftime("%Y-%m-%d") +
            "&BBOX={bbox}"
        )
        return copernicus_wms_url, scene_date
    except Exception:
        pass
        
    return None, None

def calc_geomorphology(grid, r, c, cell_size_m):
    dz_dx = (grid[r][c+1] - grid[r][c-1]) / (2 * cell_size_m)
    dz_dy = (grid[r+1][c] - grid[r-1][c]) / (2 * cell_size_m)
    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_deg = math.degrees(math.atan2(-dz_dy, dz_dx)) % 360
    
    aspect_rad = math.radians(aspect_deg)
    ideal_rad = math.radians(135.0)  # Південно-східний схил для КР
    sun_score = max(0.0, math.cos(aspect_rad - ideal_rad))
    return slope_deg, aspect_deg, round(sun_score, 2)

def detect_promontory_kr(grid, r, c):
    """
    Детектор мисових форм рельєфу спеціально для Київської Русі.
    Перевіряє наявність крутого схилу/падіння з 3+ сторін.
    """
    z = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    lower_count = 0
    max_drop = 0.0
    
    for dr, dc in dirs:
        for step in (1, 2, 3):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < rows and 0 <= nc < cols:
                drop = z - grid[nr][nc]
                if drop >= 3.5: # Вимога для КР: крутий перепад від 3.5 м
                    lower_count += 1
                    if drop > max_drop:
                        max_drop = drop
                    break
                
    if lower_count >= 3 and max_drop >= 4.0:
        return min(1.0, (lower_count / 8.0) * 0.5 + (max_drop / 15.0) * 0.5)
    return 0.0

def analyze_site_kr(grid, r, c, lat_v, lon_v, cell_size_m):
    """
    Спеціалізована функція оцінки пам'яток Київської Русі (КР)
    """
    z_center = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    
    tip_score = detect_promontory_kr(grid, r, c)
    if tip_score == 0.0:
        return None
        
    slope_deg, aspect_deg, sun_score = calc_geomorphology(grid, r, c, cell_size_m)
    
    # Пошук низини (заплава річки / струмок)
    min_z = z_center
    dist_to_water_m = 999.0
    search_r = max(3, min(10, int(450.0 / cell_size_m)))
    
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
    
    # Для КР критична висота над низиною (мінімум 6 метрів)
    if delta_h < 6.0:
        return None
        
    # Ідеальний перепад висоти для городищ/селищ КР: 12-28 метрів
    if 10.0 <= delta_h <= 30.0:
        s_height = 1.0
    elif delta_h < 10.0:
        s_height = delta_h / 10.0
    else:
        s_height = max(0.4, 1.0 - (delta_h - 30.0) / 40.0)
        
    # Дистанція до води для КР: ідеально 100 - 300 метрів
    s_water = max(0.1, 1.0 - abs(dist_to_water_m - 200.0) / 350.0)
    
    # Скоригована формула вагових коефіцієнтів для Київської Русі
    final_score = (0.45 * tip_score + 0.25 * s_height + 0.20 * s_water + 0.10 * sun_score) * 100
    
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
        "is_tip": True if tip_score >= 0.65 else False
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
    # Дрібний крок сітки (300 кроків): роздільна здатність ~25-40 метрів
    grid_steps = 300
    
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    lat_step = (2 * lat_delta) / grid_steps
    lon_step = (2 * lon_delta) / grid_steps
    cell_size_m = (2 * radius_km * 1000.0) / grid_steps
    
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
    
    for r in range(3, rows - 3):
        for c in range(3, cols - 3):
            lat_v = round(lats[r], 5)
            lon_v = round(lons[c], 5)
            res = analyze_site_kr(grid, r, c, lat_v, lon_v, cell_size_m)
            if res and res["score"] >= 42.0:
                raw_results.append(res)
                
    # Кластеризація (придшушення сусідніх точок ближче 120 м)
    clean_results = strict_nms_clustering(raw_results, min_dist_m=120.0)
    
    limit = 40 if radius_km >= 10 else 20
    return clean_results[:limit]

@app.get("/")
def read_root():
    return {"status": "GeoPredict API (Спеціалізація: Київська Русь) працює"}

@app.get("/map", response_class=HTMLResponse)
def get_map(lat: float = 50.75, lon: float = 33.47, radius_km: float = 10.0):
    # 1. Пошук кращих об'єктів Київської Русі
    results = run_analysis(lat, lon, radius_km)
    
    # 2. Отримання шару з Copernicus Data Space Ecosystem
    copernicus_wms_url, scene_date = get_copernicus_s2_ndvi_layer(lat, lon, radius_km)
    
    # 3. Створення карти
    m = folium.Map(location=[lat, lon], zoom_start=12 if radius_km > 10 else 14, tiles=None)
    
    folium.TileLayer("OpenStreetMap", name="Топо-карта (OSM)").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Супутник HD (Esri)"
    ).add_to(m)
    
    # Додавання Copernicus Sentinel-2
    folium.TileLayer(
        tiles="https://datacenter.copernicus.eu/tiles/s2/{z}/{x}/{y}.png",
        attr="Copernicus Data Space Ecosystem",
        name=f"🇪🇺 Copernicus Sentinel-2 ({scene_date})",
        overlay=True,
        opacity=0.60
    ).add_to(m)
    
    folium.Marker(
        [lat, lon],
        popup=f"Центр аналізу КР (Радіус {radius_km} км)",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)
    
    for idx, pt in enumerate(results, 1):
        if pt["score"] >= 68:
            color = "red"        # Висока ймовірність (Городище / Виражений мис КР)
        elif pt["score"] >= 52:
            color = "orange"     # Середня ймовірність (Мисове селище КР)
        else:
            color = "darkblue"   # Перспективна тераса
        
        popup_html = f"""
        <div style='font-family: sans-serif; width: 230px;'>
            <h4 style='margin:0 0 5px 0; color:#d9534f;'>Ціль КР #{idx} (Бал: {pt['score']}%)</h4>
            <b>Тип:</b> {'Оборонний мис (Городище)' if pt['is_tip'] else 'Терасове селище'}<br>
            <b>Висота над низиною:</b> +{pt['delta_h_m']} м<br>
            <b>Абс. висота:</b> {pt['elevation_m']} м<br>
            <b>До річки/заплави:</b> ~{pt['dist_water_m']} м<br>
            <b>Схил / Сонце:</b> {pt['aspect_deg']}° ({int(pt['sun_score']*100)}%)
        </div>
        """
        
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=7 if radius_km > 10 else 9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)
        
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()
