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

def score_chernyakhiv(slope, aspect, delta_h, concavity):
    s_slope = 1.0 if 2.0 <= slope <= 5.0 else (0.6 if 1.0 <= slope < 2.0 or 5.0 < slope <= 8.0 else 0.1)
    s_ravine = 1.0 if (concavity > 1.5 and 3.0 <= delta_h <= 12.0) else (0.7 if concavity > 0.5 and 2.0 <= delta_h <= 15.0 else 0.2)
    return round((0.50 * s_ravine + 0.50 * s_slope) * 100, 1)

def score_kyiv_rus(slope, aspect, delta_h, concavity):
    s_height = 1.0 if 15.0 <= delta_h <= 35.0 else (0.6 if 10.0 <= delta_h < 15.0 or 35.0 < delta_h <= 50.0 else 0.1)
    s_sun = 1.0 if 90.0 <= aspect <= 200.0 else (0.6 if 45.0 <= aspect < 90.0 or 200.0 < aspect <= 250.0 else 0.3)
    s_slope = 1.0 if 2.0 <= slope <= 7.0 else 0.4
    return round((0.45 * s_height + 0.35 * s_sun + 0.20 * s_slope) * 100, 1)

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
    results = []
    
    for r in range(2, rows - 2):
        for c in range(2, cols - 2):
            window = [grid[i][j] for i in range(r-2, r+3) for j in range(c-2, c+3)]
            z = grid[r][c]
            delta_h = z - min(window)
            concavity = (sum(window) / len(window)) - z
            slope, aspect = calc_slope_and_aspect(grid, r, c)
            
            score = score_kyiv_rus(slope, aspect, delta_h, concavity) if profile.lower() == "kyiv_rus" else score_chernyakhiv(slope, aspect, delta_h, concavity)
            
            if score >= 60.0:
                results.append({
                    "lat": round(lats[r], 5),
                    "lon": round(lons[c], 5),
                    "score": score,
                    "elevation_m": z,
                    "delta_h_m": round(delta_h, 1),
                    "slope_deg": round(slope, 1)
                })
    return results

@app.get("/")
def read_root():
    return {"status": "Сервер працює"}

@app.get("/analyze")
def analyze(lat: float, lon: float, radius_km: float = 1.0, profile: str = "chernyakhiv"):
    res = run_analysis(lat, lon, radius_km, profile)
    res.sort(key=lambda x: x["score"], reverse=True)
    return {"center": {"lat": lat, "lon": lon}, "total_hotspots": len(res), "top_results": res[:20]}

@app.get("/map", response_class=HTMLResponse)
def get_map(lat: float, lon: float, radius_km: float = 1.0, profile: str = "kyiv_rus"):
    results = run_analysis(lat, lon, radius_km, profile)
    
    # Ініціалізація карти
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    
    # Центр пошуку
    folium.Marker(
        [lat, lon],
        popup="Центр аналізу",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(m)
    
    # Відображення перспективних точок
    for pt in results:
        color = "red" if pt["score"] >= 90 else ("orange" if pt["score"] >= 75 else "green")
        popup_text = f"Бал: {pt['score']}%<br>Висота: {pt['elevation_m']}м<br>Перепад: {pt['delta_h_m']}м<br>Схил: {pt['slope_deg']}°"
        
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)
        
    return m._repr_html_()
