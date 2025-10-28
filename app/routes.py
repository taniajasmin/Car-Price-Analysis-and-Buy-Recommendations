from fastapi import APIRouter, HTTPException
from typing import List, Optional
from .models import AnalysisRequest, CarAnalysis, CompareRequest, Specs, SuggestRequest
from .ai_calculations import process_car_data, fetch_car_specs, suggest_advice
from pydantic import ValidationError, BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

@router.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Car Profit API. Use /docs for Swagger UI or /analyze-cars/, /compare-cars/, /ai-suggest/ for endpoints."}

@router.post("/analyze-cars/", response_model=List[CarAnalysis])
async def analyze_cars(request: AnalysisRequest):
    try:
        results = process_car_data(request.cars)
        return results
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-cars/", response_model=List[Specs])
async def compare_cars(request: CompareRequest):
    try:
        specs1 = fetch_car_specs(request.car1)
        specs2 = fetch_car_specs(request.car2)
        return [
            Specs(model=request.car1, **specs1),
            Specs(model=request.car2, **specs2)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/ai-suggest/")
# async def ai_suggest(request: SuggestRequest):
#     try:
#         response = suggest_advice(request.message, request.budget, request.needs)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

class ChatState(BaseModel):
    stage: Optional[int] = 0
    car_type: Optional[str] = None
    budget: Optional[float] = None
    needs: Optional[str] = None
    user_input: Optional[str] = None

@router.post("/ai-suggest/")
async def ai_suggest(chat: ChatState):
    if chat.stage == 0:
        return {
            "status": "continue",
            "stage": 1,
            "reply": "Hi there! I am your AI Car Advisor. What type of car are you looking for? (e.g., sedan, SUV, EV, family car)"
        }

    elif chat.stage == 1 and chat.user_input:
        return {
            "status": "continue",
            "stage": 2,
            "car_type": chat.user_input,
            "reply": f"Got it — you are looking for a {chat.user_input}. What’s your budget range (in USD)?"
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
            "reply": "Thanks! Lastly, any special needs? (e.g., electric, fuel efficient, low maintenance)"
        }

    elif chat.stage == 3 and chat.user_input:
        chat.needs = chat.user_input
        suggestion_text = suggest_advice(chat.car_type, chat.budget, chat.needs)
        return {"status": "complete", "reply": suggestion_text}


    else:
        return {
            "status": "error",
            "reply": "Sorry, I did not catch that. Let’s start over!"
        }