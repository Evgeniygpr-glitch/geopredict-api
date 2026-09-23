from fastapi import FastAPI
import srtm
import math

app = FastAPI()

# Ініціалізуємо SRTM
elevation_data = srtm.get_data()

@app.get("/")
def read_root():
    return {"status": "Сервер працює та готовий до роботи"}

@app.get("/elevation")
def get_elevation(lat: float, lon: float):
    altitude = elevation_data.get_elevation(lat, lon)
    return {"lat": lat, "lon": lon, "elevation_m": altitude}

@app.get("/scan_grid")
def scan_grid(lat: float, lon: float, radius_km: float = 1.0):
    """
    Формує сітку висот (DEM) у RAM у радіусі radius_km навколо точки
    з кроком ~30 метрів.
    """
    # Крок сітки ~ 30 метрів у градусах
    lat_step = 0.00027  
    lon_step = 0.00042 # з урахуванням широти ~50°
    
    # Відхилення координат для заданого радіуса
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
        
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "grid_dimensions": f"{rows}x{cols}",
        "total_points": rows * cols,
        "status": "Матрицю висот успішно завантажено в RAM"
    }
