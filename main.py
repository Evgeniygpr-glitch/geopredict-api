from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import srtm
import math
import folium

app = FastAPI()
elevation_data = srtm.get_data()

def calc_slope_and_aspect(grid, r, c, cell_size=30.0):
    dz_dx = (grid[r][c+1] - grid[r][c-1]) / (2 * cell_size)
    dz_dy = (grid[r+1][c] - grid[r-1][c]) / (2 * cell_size)
    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_deg = math.degrees(math.atan2(-dz_dy, dz_dx)) % 360
    return slope_deg, aspect_deg

def analyze_promontory_and_water(grid, r, c, cell_size=30.0):
    """
    Аналіз геометрії мису та доступності води для Київської Русі
    """
    z_center = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    
    # 1. Перевірка скиду висот у 8 напрямках на відстані ~90 м (3 пікселі)
    drops = 0
    directions = [(-3,0), (3,0), (0,-3), (0,3), (-2,-2), (-2,2), (2,-2), (2,2)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if z_center - grid[nr][nc] >= 10.0:  # перепад понад 10м
                drops += 1
                
    # Форма мису: обрив з 4-6 боків (ідеально для носа мису з ярами по боках)
    if 4 <= drops <= 6:
        s_prom = 1.0
    elif drops == 3 or drops == 7:
        s_prom = 0.6
    else:
        s_prom = 0.1  # рівне плато або прямий схил
        
    # 2. Пошук дна долини / річки в радіусі 300 м (10 пікселів)
    min_z = z_center
    dist_to_water_m = 999.0
    
    for dr in range(-10, 11):
        for dc in range(-10, 11):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                val = grid[nr][nc]
                dist_m = math.sqrt((dr * cell_size)**2 + (dc * cell_size)**2)
                if val < min_z:
                    min_z = val
                    dist_to_water_m = dist_m
                elif val == min_z and dist_m < dist_to_water_m:
                    dist_to_water_m = dist_m
                    
    delta_h = z_center - min_z  # висота над водою
    
    # Фактор відстані до води (ідеал: 50 - 250 метрів)
    if 40.0 <= dist_to_water_m <= 250.0:
        s_dist = 1.0
    elif 250.0 < dist_to_water_m <= 400.0:
        s_dist = 0.6
    else:
        s_dist = 0.2
        
    # Фактор висоти над водою (ідеал: 12 - 30 метрів)
    if 12.0 <= delta_h <= 30.0:
        s_height = 1.0
    elif 8.0 <= delta_h < 12.0 or 30.0 < delta_h <= 45.0:
        s_height = 0.6
    else:
        s_height = 0.2
        
    s_water = 0.6 * s_dist + 0.4 * s_height
    return s_prom, s_water, delta_h, round(dist_to_water_m)

def score_kyiv_rus_advanced(slope, aspect, s_prom, s_water):
    """
    Комплексна оцінка Київської Русі (Мис + Вода + Сонце + Схил)
    """
    # Сонячний схил (Пд-Сх, Пд, Сх)
    s_sun = 1.0 if 90.0 <= aspect <= 200.0 else (0.6 if 45.0 <= aspect < 90.0 or 200.0 < aspect <= 250.0 else 0.3)
    # Зручний схил верхнього плато мису
    s_slope = 1.0 if 2.0 <= slope <= 7.0 else 0.4
    
    # Підсумкова формула (ваги факторів)
    # Оборонний мис (35%) + Доступність води (35%) + Інсоляція (15%) + Зручність схилу (15%)
    score = (0.35 * s_prom + 0.35 * s_water + 0.15 * s_sun + 0.15 * s_slope) * 100
    return round(score, 1)

def filter_local_maxima(results, min_distance_m=150.0):
    """
    Кластеризація: придушення сусідніх точок у радіусі 150 м (NMS).
    Залишає лише 1 найсильніший маркер на один мис.
    """
    results.sort(key=lambda x: x["score"], reverse=True)
    filtered = []
    
    for pt in results:
        keep = True
        for existing in filtered:
            d_lat = (pt["lat"] - existing["lat"]) * 111000
            d_lon = (pt["lon"] - existing["lon"]) * 111000 * math.cos(math.radians(pt["lat"]))
            dist = math.sqrt(d_lat**2 + d_lon**2)
            
            if dist < min_distance_m:
                keep = False
                break
        if keep:
            filtered.append(pt)
            
    return filtered

def run_analysis(lat: float, lon: float, radius_km: float, profile: str):
    lat_step, lon_step = 0.00027, 0.00042
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    grid, lats, lons = [], [], []
    curr_lat = lat - lat_delta
    
    while curr_lat <= lat + lat_delta:
        lats.append(curr_lat)
        row = []
        curr_lon = lon - lon_delta
        while curr_lon <= lon + lon_delta:
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
            z = grid[r][c]
            slope, aspect = calc_slope_and_aspect(grid, r, c)
            s_prom, s_water, delta_h, dist_water = analyze_promontory_and_water(grid, r, c)
            
            score = score_kyiv_rus_advanced(slope, aspect, s_prom, s_water)
            
            if score >= 65.0:  # Поріг проходження
                raw_results.append({
                    "lat": round(lats[r], 5),
                    "lon": round(lons[c], 5),
                    "score": score,
                    "elevation_m": z,
                    "delta_h_m": round(delta_h, 1),
                    "dist_water_m": dist_water,
                    "slope_deg": round(slope, 1)
                })
                
    # Застосовуємо кластеризацію — прибираємо масиви
    clean_results = filter_local_maxima(raw_results, min_distance_m=150.0)
    return clean_results

@app.get("/")
def read_root():
    return {"status": "Сервер працює"}

@app.get("/analyze")
def analyze(lat: float, lon: float, radius_km: float = 1.0, profile: str = "kyiv_rus"):
    res = run_analysis(lat, lon, radius_km, profile)
    return {"center": {"lat": lat, "lon": lon}, "total_hotspots": len(res), "top_results": res[:20]}

@app.get("/map", response_class=HTMLResponse)
def get_map(lat: float, lon: float, radius_km: float = 1.0, profile: str = "kyiv_rus"):
    results = run_analysis(lat, lon, radius_km, profile)
    
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    
    folium.Marker(
        [lat, lon],
        popup="Центр аналізу",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)
    
    for pt in results:
        color = "red" if pt["score"] >= 85 else ("orange" if pt["score"] >= 72 else "green")
        popup_text = (f"<b>Бал КР: {pt['score']}%</b><br>"
                      f"Висота: {pt['elevation_m']}м<br>"
                      f"Перепад над водою: {pt['delta_h_m']}м<br>"
                      f"До води: ~{pt['dist_water_m']}м<br>"
                      f"Схил: {pt['slope_deg']}°")
        
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup_text
        ).add_to(m)
        
    return m._repr_html_()
