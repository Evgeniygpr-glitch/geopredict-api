from fastapi import FastAPI
import srtm
import math

app = FastAPI()
elevation_data = srtm.get_data()

def calc_slope_and_aspect(grid, r, c, cell_size=30.0):
    """Обчислення крутизни схилу та азимуту (експозиції)"""
    dz_dx = (grid[r][c+1] - grid[r][c-1]) / (2 * cell_size)
    dz_dy = (grid[r+1][c] - grid[r-1][c]) / (2 * cell_size)
    
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)
    
    aspect_rad = math.atan2(-dz_dy, dz_dx)
    aspect_deg = math.degrees(aspect_rad) % 360
    
    return slope_deg, aspect_deg

def score_chernyakhiv(slope, aspect, delta_h, concavity):
    """Профіль: Черняхівська культура (прихованість, яри, захист від вітру)"""
    # 1. Придатність схилу (ідеал 2-5 град)
    if 2.0 <= slope <= 5.0:
        s_slope = 1.0
    elif 1.0 <= slope < 2.0 or 5.0 < slope <= 8.0:
        s_slope = 0.6
    else:
        s_slope = 0.1
        
    # 2. Локальна прихованість у балочній системі (верхів'я яру)
    if concavity > 1.5 and 3.0 <= delta_h <= 12.0:
        s_ravine = 1.0
    elif concavity > 0.5 and 2.0 <= delta_h <= 15.0:
        s_ravine = 0.7
    else:
        s_ravine = 0.2
        
    # Підсумковий бал (0 - 100%)
    score = (0.50 * s_ravine + 0.50 * s_slope) * 100
    return round(score, 1)

def score_kyiv_rus(slope, aspect, delta_h, concavity):
    """Профіль: Київська Русь (високий мис, сонячний схил, огляд)"""
    # 1. Домінантна висота (ідеал 15-35 м над низом)
    if 15.0 <= delta_h <= 35.0:
        s_height = 1.0
    elif 10.0 <= delta_h < 15.0 or 35.0 < delta_h <= 50.0:
        s_height = 0.6
    else:
        s_height = 0.1
        
    # 2. Сонячна сторона (Пд-Сх, Пд, Сх: 90° - 200°)
    if 90.0 <= aspect <= 200.0:
        s_sun = 1.0
    elif 45.0 <= aspect < 90.0 or 200.0 < aspect <= 250.0:
        s_sun = 0.6
    else:
        s_sun = 0.3
        
    # 3. Крутизна схилу мису (2-7 град)
    s_slope = 1.0 if 2.0 <= slope <= 7.0 else 0.4
    
    # Підсумковий бал (0 - 100%)
    score = (0.45 * s_height + 0.35 * s_sun + 0.20 * s_slope) * 100
    return round(score, 1)

@app.get("/")
def read_root():
    return {"status": "Сервер працює та готовий до аналізу"}

@app.get("/elevation")
def get_elevation(lat: float, lon: float):
    altitude = elevation_data.get_elevation(lat, lon)
    return {"lat": lat, "lon": lon, "elevation_m": altitude}

@app.get("/scan_grid")
def scan_grid(lat: float, lon: float, radius_km: float = 1.0):
    lat_step, lon_step = 0.00027, 0.00042
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    grid = []
    curr_lat = lat - lat_delta
    while curr_lat <= lat + lat_delta:
        row = []
        curr_lon = lon - lon_delta
        while curr_lon <= lon + lon_delta:
            alt = elevation_data.get_elevation(curr_lat, curr_lon)
            row.append(alt if alt is not None else 0)
            curr_lon += lon_step
        grid.append(row)
        curr_lat += lat_step
        
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "grid_dimensions": f"{len(grid)}x{len(grid[0]) if grid else 0}",
        "total_points": len(grid) * (len(grid[0]) if grid else 0),
        "status": "Матрицю висот успішно завантажено в RAM"
    }

@app.get("/analyze")
def analyze_area(lat: float, lon: float, radius_km: float = 1.0, profile: str = "chernyakhiv"):
    lat_step, lon_step = 0.00027, 0.00042
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    grid = []
    lats, lons = [], []
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
    
    # Розрахунок параметрів для точок сітки
    for r in range(2, rows - 2):
        for c in range(2, cols - 2):
            cell_lat = lats[r]
            cell_lon = lons[c]
            z = grid[r][c]
            
            # Набір точок у вікні 5x5 навколо поточного пікселя
            window = [grid[i][j] for i in range(r-2, r+3) for j in range(c-2, c+3)]
            z_min = min(window)
            z_mean = sum(window) / len(window)
            
            delta_h = z - z_min
            concavity = z_mean - z
            slope, aspect = calc_slope_and_aspect(grid, r, c)
            
            if profile.lower() == "kyiv_rus":
                score = score_kyiv_rus(slope, aspect, delta_h, concavity)
            else:
                score = score_chernyakhiv(slope, aspect, delta_h, concavity)
                
            if score >= 60.0:
                results.append({
                    "lat": round(cell_lat, 5),
                    "lon": round(cell_lon, 5),
                    "score": score,
                    "elevation_m": z,
                    "delta_h_m": round(delta_h, 1),
                    "slope_deg": round(slope, 1),
                    "aspect_deg": round(aspect, 1)
                })
                
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "profile": profile,
        "total_hotspots_found": len(results),
        "top_results": results[:15]
    }
