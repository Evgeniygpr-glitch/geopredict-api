from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import folium
import pystac_client
import planetary_computer

app = FastAPI()

def get_harvested_fields_ndvi_layer(lat: float, lon: float):
    """
    Шукає найновіший знімок Sentinel-2 L2A та генерує URL тайлового шару NDVI, 
    відкаліброваного саме під скошені поля (NDVI 0.10 - 0.35).
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Пошук останнього знімка з мінімальною хмарністю (<15%)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        max_items=1,
        query={"eo:cloud_cover": {"lt": 15}}
    )
    
    items = list(search.items())
    if not items:
        return None, None

    item = items[0]
    date_str = item.datetime.strftime("%Y-%m-%d")

    # Dynamic Titiler URL з виразом NDVI та діапазоном 0.10-0.35 (скошені поля)
    tile_url = (
        f"https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x"
        f"?collection=sentinel-2-l2a&item={item.id}"
        f"&expression=(B08-B04)/(B08+B04)"
        f"&rescale=0.10,0.35"
        f"&colormap_name=YlOrRd"  # Жовто-помаранчево-червоний градієнт для стерні
    )
    
    return tile_url, date_str


@app.get("/map", response_class=HTMLResponse)
def generate_map(lat: float = 50.75, lon: float = 33.47):
    # Створення базової карти
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")

    # Супутникова підкладка Esri World Imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite'
    ).add_to(m)

    # Отримання тайлів Sentinel-2 NDVI для скошених полів
    ndvi_tile_url, date_str = get_harvested_fields_ndvi_layer(lat, lon)

    if ndvi_tile_url:
        folium.TileLayer(
            tiles=ndvi_tile_url,
            attr="Copernicus Sentinel-2 / Microsoft Planetary Computer",
            name=f"🌾 Скошені поля NDVI (Sentinel-2, {date_str})",
            overlay=True,
            opacity=0.75
        ).add_to(m)

    # Панель перемикання шарів
    folium.LayerControl(collapsed=False).add_to(m)

    return m._repr_html_()
