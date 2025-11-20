from fastapi import FastAPI, HTTPException, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import router as api_router
from pydantic import BaseModel
from typing import List, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI
# import openai
import os

from app.ai_calculations import process_car_data, fetch_car_specs, load_car_data
from app.routes import router as routes_router


app = FastAPI(title="Car Profit and Analysis API")

router = APIRouter()

app.include_router(routes_router)
app.include_router(router)

# Load environment variables and set up OpenAI
load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Model for Car Input
class CarInput(BaseModel):
    title: str
    brand: Optional[str] = None
    year_numeric: Optional[float] = None
    mileage_numeric: Optional[float] = None
    price_numeric: Optional[float] = None
    is_premium: Optional[int] = 0
    age: Optional[float] = 0

# Root Endpoint
def home():
    return {"message": "Car Price Compare & AI Advisor API is running!"}

# Car name and image
@app.get("/cars/list")
async def list_cars():
    """
    Returns raw scraped cars with title, image_url, url, and price.
    Useful for displaying car cards in frontend.
    """
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


# Analyze Cars Route
@app.post("/analyze-cars/")
async def analyze_cars(payload: Any = Body(...)):
    """
    Accepts:
      - a JSON array of car objects, OR
      - {"cars": [ ... ] }
    Returns processed car list and summary.
    """
    try:
        if isinstance(payload, dict) and "cars" in payload:
            cars_list = payload["cars"]
        elif isinstance(payload, list):
            cars_list = payload
        else:
            raise HTTPException(
                status_code=400,
                detail='Invalid payload. Send a JSON array or {"cars": [...]}',
            )

        processed = process_car_data(cars_list)

        summary = {
            "total_cars": len(processed),
            "avg_profit": sum([c.get("expected_profit", 0) for c in processed])
            / len(processed)
            if processed
            else 0,
            "buy_recommendations": [
                c.get("title")
                for c in processed
                if "BUY" in str(c.get("recommendation", "")).upper()
            ],
        }

        return JSONResponse(
            content={"status": "success", "data": processed, "summary": summary}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Compare Cars Route
@app.post("/compare-cars/")
async def compare_cars(payload: Any = Body(...)):
    try:
        if isinstance(payload, list):
            processed = process_car_data(payload)
            return JSONResponse(content={"status": "success", "data": processed})

        if isinstance(payload, dict) and ("car1" in payload or "car2" in payload):
            car1 = payload.get("car1")
            car2 = payload.get("car2")
            specs1 = fetch_car_specs(car1) if car1 else {}
            specs2 = fetch_car_specs(car2) if car2 else {}
            return JSONResponse(
                content={"status": "success", "data": {"car1": specs1, "car2": specs2}}
            )

        if isinstance(payload, dict) and "cars" in payload:
            processed = process_car_data(payload["cars"])
            return JSONResponse(content={"status": "success", "data": processed})

        raise HTTPException(
            status_code=400,
            detail="Invalid payload for compare-cars. Send array of objects or {'car1':name,'car2':name}",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# AI Suggestion Logic
def suggest_advice(message, budget=None, needs=None):
    """
    Generates AI-based car buying advice using OpenAI API.
    """
    try:
        prompt = (
            f"User query: {message}. "
            f"Budget: {budget or 'not specified'}. "
            f"Needs: {needs or 'not specified'}. "
            f"Provide friendly and helpful car-buying advice based on market trends."
        )

        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional car advisor who helps people "
                        "choose cars based on budget, lifestyle, and market value. "
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
        print(f"AI Error: {e}")
        return "Sorry, I’m having trouble fetching AI advice right now."

class ChatState(BaseModel):
    stage: int
    user_input: str | None = None
    car_type: str | None = None
    budget: float | None = None
    needs: str | None = None

def suggest_advice(message, budget=None, needs=None):
    try:
        prompt = (
            f"User query: {message}. "
            f"Budget: {budget or 'not specified'}. "
            f"Needs: {needs or 'not specified'}. "
            f"Give friendly, realistic car buying advice based on today's market."
        )

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional car advisor. Give clear, practical recommendations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200
        )

        # return response.choices[0].message["content"]
        return response.choices[0].message.content

    except Exception as e:
        print("AI ERROR:", e)
        return "Sorry, I’m having trouble fetching AI advice right now."

@app.post("/ai-suggest/")
async def ai_suggest(chat: ChatState):

    # STAGE 0
    if chat.stage == 0:
        return {
            "status": "continue",
            "stage": 1,
            "reply": "Hi there! I’m your AI Car Advisor. What type of car are you looking for? (e.g., sedan, SUV, EV, family car)",
        }

    # STAGE 1
    elif chat.stage == 1 and chat.user_input:
        chat.car_type = chat.user_input
        return {
            "status": "continue",
            "stage": 2,
            "car_type": chat.car_type,
            "reply": f"Great choice! What's your budget range (in USD)?",
        }

    # STAGE 2
    elif chat.stage == 2 and chat.user_input:
        try:
            chat.budget = float(chat.user_input.replace("$", "").replace(",", ""))
        except:
            chat.budget = None

        return {
            "status": "continue",
            "stage": 3,
            "budget": chat.budget,
            "reply": "Thanks! Lastly, do you have any specific needs? (electric, low maintenance, fuel efficiency, etc.)",
        }

    # STAGE 3
    elif chat.stage == 3 and chat.user_input:
        chat.needs = chat.user_input

        suggestion_text = suggest_advice(
            message=f"Car type: {chat.car_type}",
            budget=chat.budget,
            needs=chat.needs,
        )

        return {"status": "complete", "reply": suggestion_text}

    # ERROR
    else:
        return {
            "status": "error",
            "reply": "Sorry, I didn’t catch that. Let’s start over!",
        }
