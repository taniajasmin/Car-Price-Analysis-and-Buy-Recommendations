from fastapi import APIRouter, HTTPException
from typing import List
from .models import AnalysisRequest, CarAnalysis, CompareRequest, Specs, SuggestRequest
from .ai_calculations import process_car_data, fetch_car_specs, suggest_advice
from pydantic import ValidationError
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

@router.post("/ai-suggest/")
async def ai_suggest(request: SuggestRequest):
    try:
        response = suggest_advice(request.message, request.budget, request.needs)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))