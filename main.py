from fastapi import FastAPI
import srtm

app = FastAPI()

# Ініціалізуємо завантажувач рельєфу SRTM
elevation_data = srtm.get_data()

@app.get("/")
def read_root():
    return {"status": "Сервер працює та готовий до роботи"}

@app.get("/elevation")
def get_elevation(lat: float, lon: float):
    # Отримуємо висоту над рівнем моря (в метрах) для вказаних координат
    altitude = elevation_data.get_elevation(lat, lon)
    return {
        "lat": lat,
        "lon": lon,
        "elevation_m": altitude
    }
