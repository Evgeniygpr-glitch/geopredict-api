"""
Одноразово збирає список населених пунктів Сумської області з OpenStreetMap
і зберігає його в data/places_sumy.json (у репозиторії поруч із main.py).

Запуск на вашому комп'ютері (Render для цього не потрібен):
    pip install requests
    python tools/build_places.py
"""
import json
import os
import sys

import requests

# south, west, north, east — з запасом, трохи ширше за Сумщину
BBOX = "49.9,32.4,52.2,35.8"
QUERY = (
    "[out:json][timeout:180];"
    'nwr["place"~"^(city|town|village|hamlet|isolated_dwelling)$"]["name"]'
    f"({BBOX});out center;"
)
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
UA = "GeoPredict-build/1.0 (archaeology research helper)"

resp = None
for url in MIRRORS:
    try:
        print(f"Запит до {url} ...")
        resp = requests.post(url, data={"data": QUERY}, timeout=240, headers={"User-Agent": UA})
        resp.raise_for_status()
        break
    except Exception as exc:
        print(f"  не вдалось: {exc}")
        resp = None
if resp is None:
    sys.exit("Усі дзеркала Overpass недоступні. Спробуйте пізніше.")

places = []
for el in resp.json().get("elements", []):
    t = el.get("tags", {})
    lat = el.get("lat") or (el.get("center") or {}).get("lat")
    lon = el.get("lon") or (el.get("center") or {}).get("lon")
    name = t.get("name:uk") or t.get("name")
    if name and lat is not None and lon is not None:
        places.append({"name": name, "kind": t.get("place", ""),
                       "lat": round(lat, 5), "lon": round(lon, 5)})

os.makedirs("data", exist_ok=True)
with open(os.path.join("data", "places_sumy.json"), "w", encoding="utf-8") as f:
    json.dump({"places": places}, f, ensure_ascii=False)
print(f"Збережено {len(places)} населених пунктів у data/places_sumy.json")
