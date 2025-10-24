from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any
from app.ai_calculations import process_car_data, fetch_car_specs, suggest_advice

app = FastAPI(title="Car Profit and Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CarInput(BaseModel):
    title: str
    brand: Optional[str] = None
    year_numeric: Optional[float] = None
    mileage_numeric: Optional[float] = None
    price_numeric: Optional[float] = None
    is_premium: Optional[int] = 0
    age: Optional[float] = 0


@app.post("/analyze-cars/")
async def analyze_cars(payload: Any = Body(...)):
    """
    Accepts either:
      - a JSON array of car objects (preferred), OR
      - {"cars": [ ... ] }.
    Returns processed car list and a summary.
    """
    try:
        if isinstance(payload, dict) and "cars" in payload:
            cars_list = payload["cars"]
        elif isinstance(payload, list):
            cars_list = payload
        else:
            raise HTTPException(status_code=400, detail="Invalid payload. Send a JSON array or {\"cars\": [...]}")

        processed = process_car_data(cars_list)

        summary = {
            "total_cars": len(processed),
            "avg_profit": sum([c.get("expected_profit", 0) for c in processed]) / len(processed) if processed else 0,
            # only include titles/identifiers to avoid duplicating objects
            "buy_recommendations": [c.get("title") for c in processed if "BUY" in str(c.get("recommendation", "")).upper()]
        }

        return JSONResponse(content={"status": "success", "data": processed, "summary": summary})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-cars/")
async def compare_cars(payload: Any = Body(...)):
    """
    Two modes:
    1) If body is array of car objects -> returns processed results (same as analyze).
    2) If body is {"car1": "name", "car2": "name"} -> returns specs from CSV.
    """
    try:
        # Mode 1: array -> process and return JSON
        if isinstance(payload, list):
            processed = process_car_data(payload)
            return JSONResponse(content={"status": "success", "data": processed})

        # Mode 2: named strings -> fetch specs
        if isinstance(payload, dict) and ("car1" in payload or "car2" in payload):
            car1 = payload.get("car1")
            car2 = payload.get("car2")
            specs1 = fetch_car_specs(car1) if car1 else {}
            specs2 = fetch_car_specs(car2) if car2 else {}
            return JSONResponse(content={"status": "success", "data": {"car1": specs1, "car2": specs2}})

        # Mode 3: { "cars": [...] } wrapper
        if isinstance(payload, dict) and "cars" in payload:
            processed = process_car_data(payload["cars"])
            return JSONResponse(content={"status": "success", "data": processed})

        raise HTTPException(status_code=400, detail="Invalid payload for compare-cars. Send array of objects or {'car1':name,'car2':name}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/ai-suggest/")
# async def ai_suggest(data: dict = Body(...)):
#     try:
#         prompt = data.get("prompt")
#         budget = data.get("budget")
#         needs = data.get("needs")
#         if not prompt:
#             raise HTTPException(status_code=400, detail="Missing 'prompt' in request body.")
#         reply = suggest_advice(prompt, budget, needs)
#         return JSONResponse(content={"status": "success", "reply": reply})
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/ai-suggest/")
async def ai_suggest(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt") or data.get("message")  # ✅ accept both
        budget = data.get("budget")
        needs = data.get("needs")

        if not prompt:
            return JSONResponse(status_code=400, content={"error": "Missing prompt or message"})

        suggestion = suggest_advice(prompt, budget, needs)
        return {"status": "success", "reply": suggestion}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    


@app.get("/car-specs/{model}")
async def get_car_specs(model: str):
    try:
        specs = fetch_car_specs(model)
        return JSONResponse(content={"status": "success", "specs": specs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
