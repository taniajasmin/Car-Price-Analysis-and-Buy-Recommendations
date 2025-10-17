from pydantic import BaseModel
from typing import List, Optional

class CarData(BaseModel):
    title: str
    url: str
    price_numeric: Optional[float] = None
    year_numeric: Optional[int] = None
    mileage_numeric: Optional[int] = None
    brand: Optional[str] = None
    age: Optional[int] = None
    is_premium: Optional[bool] = False

class CarAnalysis(BaseModel):
    title: str
    url: str
    estimated_market_value: float
    total_costs: float
    expected_profit: float
    profit_margin_pct: float
    risk_level: str
    recommendation: str

class AnalysisRequest(BaseModel):
    cars: List[CarData]

class Specs(BaseModel):
    model: str
    engine: Optional[str] = None
    fuel_type: Optional[str] = None
    max_power: Optional[str] = None
    driving_range: Optional[str] = None
    drivetrain: Optional[str] = None
    price: Optional[float] = None  # From input or estimated

class CompareRequest(BaseModel):
    car1: str  
    car2: str  

class SuggestRequest(BaseModel):
    message: str  # User chat message
    budget: Optional[float] = None
    needs: Optional[str] = None  # e.g., "family"