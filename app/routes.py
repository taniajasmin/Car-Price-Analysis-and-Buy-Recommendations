from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import ValidationError
from .models import AnalysisRequest, CarAnalysis, CompareRequest, Specs
from .ai_calculations import process_car_data, fetch_car_specs
import json
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to Car Profit API",
        "usage": "Use /docs for Swagger UI or POST /analyze-cars/ and /compare-cars/."
    }


@router.post("/analyze-cars/", response_model=List[CarAnalysis])
async def analyze_cars(request: AnalysisRequest):
    """
    Analyze a list of cars and return profit, risk, and recommendations.
    """
    try:
        results = process_car_data(request.cars)
        return results
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-cars/", response_model=List[Specs])
async def compare_cars(request: CompareRequest):
    """
    Compare two cars by fetching their specifications from cars_data.json.
    """
    try:
        specs1 = fetch_car_specs(request.car1)
        specs2 = fetch_car_specs(request.car2)
        return [Specs(**specs1), Specs(**specs2)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
