from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import folium

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>GeoPredict Mini</title></head>
        <body style="font-family: sans-serif; padding: 20px; text-align: center;">
            <h2>🗺️ GeoPredict — Пошук точок</h2>
            <form action="/map" method="get">
                <label>Широта (Lat):</label><br>
                <input type="text" name="lat" value="50.6885" style="font-size: 18px; margin: 5px;"><br><br>
                <label>Довгота (Lon):</label><br>
                <input type="text" name="lon" value="33.3980" style="font-size: 18px; margin: 5px;"><br><br>
                <button type="submit" style="font-size: 20px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px;">Показати карту</button>
            </form>
        </body>
    </html>
    """

@app.get("/map", response_class=HTMLResponse)
def generate_map(lat: float = Query(50.6885), lon: float = Query(33.3980)):
    # Створюємо карту Leaflet з центром у ваших координатах
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")
    
    # Додаємо маркер вашого центру
    folium.Marker(
        [lat, lon], 
        popup="Центр пошуку", 
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Приклад: Сервер розрахував тестову перспективну точку поруч
    test_target_lat = lat + 0.005
    test_target_lon = lon + 0.008
    
    folium.CircleMarker(
        location=[test_target_lat, test_target_lon],
        radius=10,
        popup="Перспективне місце (Черняхи) - Бал: 85%",
        color="green",
        fill=True,
        fill_color="lime"
    ).add_to(m)

    # Повертаємо готову HTML-сторінку карти
    return m._repr_html_()
