from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Сервер для аналізу рельєфу працює!"}

@app.get("/health")
def health():
    return {"status": "healthy"}
