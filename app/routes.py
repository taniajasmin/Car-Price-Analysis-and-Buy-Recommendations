from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import ValidationError, BaseModel
from .models import AnalysisRequest, CarAnalysis, CompareRequest, Specs
from .ai_calculations import process_car_data, fetch_car_specs
import openai
import os
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

@router.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to Car Profit API",
        "usage": "Use /docs for Swagger UI or POST /analyze-cars/ and /compare-cars/."
    }

# ANALYZE CARS
@router.post("/analyze-cars/", response_model=List[CarAnalysis])
async def analyze_cars(request: AnalysisRequest):
    """
    Analyze car list → return profit, price score, suggestions, risk score.
    """
    try:
        results = process_car_data(request.cars)
        return results
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# COMPARE TWO CARS
@router.post("/compare-cars/", response_model=List[Specs])
async def compare_cars(request: CompareRequest):
    """
    Compare two car models using cars_data.json specs.
    """
    try:
        specs1 = fetch_car_specs(request.car1)
        specs2 = fetch_car_specs(request.car2)
        return [Specs(**specs1), Specs(**specs2)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI CHAT ENDPOINT
class ChatRequest(BaseModel):
    message: str

@router.post("/ai/chat")
async def ai_chat(request: ChatRequest):
    """
    Simple AI chat endpoint that forwards user message to OpenAI GPT model.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": request.message}]
        )

        reply = response.choices[0].message["content"]
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
