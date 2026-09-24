from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import srtm
import math
import folium

app = FastAPI()
elevation_data = srtm.get_data()

def calc_geomorphology(grid, r, c, cell_size=30.0):
    """
    Обчислення схилу, азимуту (aspect) та коефіцієнта сонячної освітленості
    """
    dz_dx = (grid[r][c+1] - grid[r][c-1]) / (2 * cell_size)
    dz_dy = (grid[r+1][c] - grid[r-1][c]) / (2 * cell_size)
    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_deg = math.degrees(math.atan2(-dz_dy, dz_dx)) % 360
    
    # Розрахунок освітленості (ідеал: Пд-Сх / Пд від 110° до 160°)
    # 135° (Пд-Сх) отримує 1.0, Північ (0°/360°) отримує 0.0
    aspect_rad = math.radians(aspect_deg)
    ideal_rad = math.radians(135.0) # Оптимальне ранкове/денне сонце
    
    sun_score = max(0.0, math.cos(aspect_rad - ideal_rad))
    return slope_deg, aspect_deg, round(sun_score, 2)

def detect_promontory_tip(grid, r, c):
    """
    Детектор КІНЧИКА (носа) мису:
    Шукає точку, де з 3-х боків обрив/спад, а з 1-го боку — плато.
    """
    z = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    
    # 8 напрямків: Пн, Пд, Зах, Сх, і 4 діагоналі (крок ~60м / 2 пікселі)
    dirs = [(-2,0), (2,0), (0,-2), (0,2), (-2,-2), (-2,2), (2,-2), (2,2)]
    lower_sectors = 0
    
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if z - grid[nr][nc] >= 8.0: # Спад висоти на 8+ метрів
                lower_sectors += 1
                
    # Край мису: 4, 5 або 6 секторів знизу (ідеальний ніс)
    if 4 <= lower_sectors <= 6:
        return 1.0  # Це чіткий ніс мису
    elif lower_sectors == 3:
        return 0.5  # Бічна кромка
    else:
        return 0.0  # Рівне поле або глибокий яр

def analyze_site_precision(grid, r, c, cell_size=30.0):
    z_center = grid[r][c]
    rows, cols = len(grid), len(grid[0])
    
    slope_deg, aspect_deg, sun_score = calc_geomorphology(grid, r, c, cell_size)
    tip_score = detect_promontory_tip(grid, r, c)
    
    if tip_score == 0.0:
        return None  # Відсікаємо все, що не є мисом/краєм
        
    # Пошук дна водотоку в радіусі 300м
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
                    
    delta_h = z_center - min_z
    
    # Жорсткі критерії КР:
    # 1. Перепад висоти над водою (12-28 м)
    s_height = 1.0 if 12.0 <= delta_h <= 28.0 else (0.5 if 8.0 <= delta_h < 12.0 or 28.0 < delta_h <= 40.0 else 0.0)
    # 2. Відстань до джерела/річки (60-220 м)
    s_water = 1.0 if 60.0 <= dist_to_water_m <= 220.0 else (0.5 if 220.0 < dist_to_water_m <= 350.0 else 0.0)
    # 3. Зручність майданчика (1.5° - 6.0°)
    s_slope = 1.0 if 1.5 <= slope_deg <= 6.0 else 0.3
    
    if s_height == 0.0 or s_water == 0.0:
        return None
        
    # Точний підсумковий бал (Ваги: Край мису 35%, Вода 25%, Висота 20%, Сонячність 20%)
    total_score = (0.35 * tip_score + 0.25 * s_water + 0.20 * s_height + 0.20 * sun_score) * 100
    
    return {
        "lat": 0.0, # заповниться вище
        "lon": 0.0,
        "score": round(total_score, 1),
        "elevation_m": z_center,
        "delta_h_m": round(delta_h, 1),
        "dist_water_m": round(dist_to_water_m),
        "slope_deg": round(slope_deg, 1),
        "aspect_deg": round(aspect_deg, 1),
        "sun_score": sun_score,
        "is_tip": True if tip_score == 1.0 else False
    }

def strict_nms_clustering(results, min_distance_m=250.0):
    """
    Жорстке придушення сусідніх точок у радіусі 250 м.
    Залишає ЛИШЕ 1 еталонну точку на локацію.
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

def run_analysis(lat: float, lon: float, radius_km: float):
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
            res = analyze_site_precision(grid, r, c)
            if res and res["score"] >= 70.0:
                res["lat"] = round(lats[r], 5)
                res["lon"] = round(lons[c], 5)
                raw_results.append(res)
                
    # Застосовуємо жорсткий NMS
    clean_results = strict_nms_clustering(raw_results, min_distance_m=250.0)
    return clean_results[:7]  # Видаємо максимум ТОР-7 точкових цілей

@app.get("/")
def read_root():
    return {"status": "Сервер працює"}

@app.get("/analyze")
def analyze(lat: float, lon: float, radius_km: float = 1.5):
    res = run_analysis(lat, lon, radius_km)
    return {"center": {"lat": lat, "lon": lon}, "total_hotspots": len(res), "top_results": res}

@app.get("/map", response_class=HTMLResponse)
def get_map(lat: float, lon: float, radius_km: float = 1.5):
    results = run_analysis(lat, lon, radius_km)
    
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    
    folium.Marker(
        [lat, lon],
        popup="Центр аналізу",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)
    
    for idx, pt in enumerate(results, 1):
        color = "red" if pt["score"] >= 85 else "orange"
        
        # Наочна картка точкової цілі
        popup_html = f"""
        <div style='font-family: sans-serif; width: 180px;'>
            <h4 style='margin:0 0 5px 0; color:#d9534f;'>Ціль #{idx} (Бал: {pt['score']}%)</h4>
            <b>Тип:</b> {'Ніс мису (Край)' if pt['is_tip'] else 'Кромка тераси'}<br>
            <b>Освітлення:</b> {pt['aspect_deg']}° (Сонце: {int(pt['sun_score']*100)}%)<br>
            <b>Висота:</b> {pt['elevation_m']} м<br>
            <b>Перепад:</b> {pt['delta_h_m']} м над низом<br>
            <b>До води:</b> ~{pt['dist_water_m']} м<br>
            <b>Схил:</b> {pt['slope_deg']}°
        </div>
        """
        
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=220)
        ).add_to(m)
        
    return m._repr_html_()
