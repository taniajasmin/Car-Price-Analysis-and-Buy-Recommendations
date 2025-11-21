from fastapi import FastAPI, HTTPException, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
from dotenv import load_dotenv
import os

from openai import OpenAI

from app.ai_calculations import (
    process_car_data,
    fetch_car_specs,
    load_car_data
)

app = FastAPI(title="Car Profit and Analysis API")

router = APIRouter()

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found! Create .env file with your key.")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatState(BaseModel):
    stage: Optional[int] = 0
    car_type: Optional[str] = None
    budget: Optional[float] = None
    needs: Optional[str] = None
    user_input: Optional[str] = None

@app.get("/")
def home():
    return {"message": "Car Price Compare & AI Advisor API is running!"}


# Get Scraped Car List (Name + Image + Price)
@app.get("/cars/list")
async def list_cars():
    try:
        cars = load_car_data()
        if not cars:
            return JSONResponse(
                content={"status": "error", "message": "No car data found."},
                status_code=404
            )

        output = []
        for c in cars:
            output.append({
                "title": c.get("title"),
                "image_url": c.get("image_url"),
                "url": c.get("url"),
                "price": c.get("price"),
                "brand": c.get("brand"),
                "year_numeric": c.get("year_numeric"),
                "mileage_numeric": c.get("mileage_numeric"),
            })

        return {"status": "success", "cars": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Analyze Cars
@app.post("/analyze-cars/", response_model=None)
async def analyze_cars(payload: Any = Body(...)):
    try:
        if isinstance(payload, dict) and "cars" in payload:
            cars_list = payload["cars"]
        elif isinstance(payload, list):
            cars_list = payload
        else:
            raise HTTPException(
                status_code=400,
                detail='Invalid payload. Send a JSON array or {"cars": [...]}'
            )

        processed = process_car_data(cars_list)

        summary = {
            "total_cars": len(processed),
            "avg_profit": (
                sum([c.get("expected_profit", 0) for c in processed]) / len(processed)
                if processed else 0
            ),
            # "buy_recommendations": [
            #     c.get("title")
            #     for c in processed
            #     if "BUY" in str(c.get("recommendation", "")).upper()
            # ],
            "buy_recommendations": [
                c.get("title")
                for c in processed
                if c.get("recommendation") in ["BUY", "STRONG BUY"]
            ],
        }

        return JSONResponse(content={"status": "success", "data": processed, "summary": summary})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Compare Cars
@app.post("/compare-cars/", response_model=None)
async def compare_cars(payload: Any = Body(...)):
    try:
        # Mode 1: car1 + car2 as names
        if isinstance(payload, dict) and ("car1" in payload and "car2" in payload):
            car1 = payload.get("car1")
            car2 = payload.get("car2")
            specs1 = fetch_car_specs(car1)
            specs2 = fetch_car_specs(car2)
            return {"status": "success", "data": {"car1": specs1, "car2": specs2}}

        # Mode 2: array of car objects
        if isinstance(payload, list):
            processed = process_car_data(payload)
            return {"status": "success", "data": processed}

        # Mode 3: {"cars": [...]}
        if isinstance(payload, dict) and "cars" in payload:
            processed = process_car_data(payload["cars"])
            return {"status": "success", "data": processed}

        raise HTTPException(
            status_code=400,
            detail="Invalid payload format for compare-cars."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# AI Suggest (Chat)
def suggest_advice(message, budget=None, needs=None):
    prompt = (
        f"User query: {message}. "
        f"Budget: {budget or 'not specified'}. "
        f"Needs: {needs or 'not specified'}. "
        f"Provide friendly and helpful car-buying advice."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional car advisor. "
                        "Give concise, warm, and practical recommendations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Error retrieving AI advice."


@app.post("/ai-suggest/")
async def ai_suggest(chat: ChatState):
    if chat.stage == 0:
        return {
            "status": "continue",
            "stage": 1,
            "reply": "Hi! What type of car are you looking for? (sedan, SUV, EV, family car?)"
        }

    elif chat.stage == 1 and chat.user_input:
        return {
            "status": "continue",
            "stage": 2,
            "car_type": chat.user_input,
            "reply": f"Nice choice. What is your budget for a {chat.user_input}?"
        }

    elif chat.stage == 2 and chat.user_input:
        try:
            budget = float(chat.user_input.replace("$", "").replace(",", ""))
        except:
            budget = None

        return {
            "status": "continue",
            "stage": 3,
            "budget": budget,
            "reply": "Got it. Any special needs? (electric, fuel efficient, low maintenance)"
        }

    elif chat.stage == 3 and chat.user_input:
        advice = suggest_advice(chat.car_type, chat.budget, chat.user_input)
        return {"status": "complete", "reply": advice}

    return {"status": "error", "reply": "Invalid state. Please restart."}


# Car Specs Lookup
@app.get("/car-specs/{model}")
async def get_car_specs(model: str):
    try:
        specs = fetch_car_specs(model)
        return {"status": "success", "specs": specs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
