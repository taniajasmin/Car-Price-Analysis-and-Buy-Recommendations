# from fastapi import FastAPI, HTTPException, Request, Body, APIRouter
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import List, Optional, Any
# from app.ai_calculations import process_car_data, fetch_car_specs, suggest_advice
# from app.routes import router as routes_router

# app = FastAPI(title="Car Profit and Analysis API")
# app.include_router(routes_router)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class CarInput(BaseModel):
#     title: str
#     brand: Optional[str] = None
#     year_numeric: Optional[float] = None
#     mileage_numeric: Optional[float] = None
#     price_numeric: Optional[float] = None
#     is_premium: Optional[int] = 0
#     age: Optional[float] = 0

# @app.get("/")
# def home():
#     return {"message": "Car Price Compare API is running!"}


# @app.post("/analyze-cars/")
# async def analyze_cars(payload: Any = Body(...)):
#     """
#     Accepts either:
#       - a JSON array of car objects (preferred), OR
#       - {"cars": [ ... ] }.
#     Returns processed car list and a summary.
#     """
#     try:
#         if isinstance(payload, dict) and "cars" in payload:
#             cars_list = payload["cars"]
#         elif isinstance(payload, list):
#             cars_list = payload
#         else:
#             raise HTTPException(status_code=400, detail="Invalid payload. Send a JSON array or {\"cars\": [...]}")

#         processed = process_car_data(cars_list)

#         summary = {
#             "total_cars": len(processed),
#             "avg_profit": sum([c.get("expected_profit", 0) for c in processed]) / len(processed) if processed else 0,
#             # only include titles/identifiers to avoid duplicating objects
#             "buy_recommendations": [c.get("title") for c in processed if "BUY" in str(c.get("recommendation", "")).upper()]
#         }

#         return JSONResponse(content={"status": "success", "data": processed, "summary": summary})

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/compare-cars/")
# async def compare_cars(payload: Any = Body(...)):
#     try:
#         # array -> process and return JSON
#         if isinstance(payload, list):
#             processed = process_car_data(payload)
#             return JSONResponse(content={"status": "success", "data": processed})

#         # named strings -> fetch specs
#         if isinstance(payload, dict) and ("car1" in payload or "car2" in payload):
#             car1 = payload.get("car1")
#             car2 = payload.get("car2")
#             specs1 = fetch_car_specs(car1) if car1 else {}
#             specs2 = fetch_car_specs(car2) if car2 else {}
#             return JSONResponse(content={"status": "success", "data": {"car1": specs1, "car2": specs2}})

#         # { "cars": [...] } wrapper
#         if isinstance(payload, dict) and "cars" in payload:
#             processed = process_car_data(payload["cars"])
#             return JSONResponse(content={"status": "success", "data": processed})

#         raise HTTPException(status_code=400, detail="Invalid payload for compare-cars. Send array of objects or {'car1':name,'car2':name}")

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=700, detail=str(e))


# # @app.post("/ai-suggest/")
# # async def ai_suggest(request: Request):
# #     try:
# #         data = await request.json()
# #         prompt = data.get("prompt") or data.get("message")  # accept both
# #         budget = data.get("budget")
# #         needs = data.get("needs")

# #         if not prompt:
# #             return JSONResponse(status_code=400, content={"error": "Missing prompt or message"})

# #         suggestion = suggest_advice(prompt, budget, needs)
# #         return {"status": "success", "reply": suggestion}

# #     except Exception as e:
# #         return JSONResponse(status_code=500, content={"error": str(e)})

# router = APIRouter()

# class ChatState(BaseModel):
#     stage: Optional[int] = 0
#     car_type: Optional[str] = None
#     budget: Optional[float] = None
#     needs: Optional[str] = None
#     user_input: Optional[str] = None

# @router.post("/ai-suggest/")
# async def ai_suggest(chat: ChatState):
#     # Greeting
#     if chat.stage == 0:
#         return {
#             "status": "continue",
#             "stage": 1,
#             "reply": "Hi there! I’m your AI Car Advisor. What type of car are you looking for? (e.g., sedan, SUV, EV, family car)"
#         }

#     # Ask for budget
#     elif chat.stage == 1 and chat.user_input:
#         return {
#             "status": "continue",
#             "stage": 2,
#             "car_type": chat.user_input,
#             "reply": f"Got it — you’re looking for a {chat.user_input}. What’s your budget range (in USD or EUR)?"
#         }

#     # Ask for specific needs
#     elif chat.stage == 2 and chat.user_input:
#         try:
#             budget = float(chat.user_input.replace("$", "").replace(",", ""))
#         except:
#             budget = None
#         return {
#             "status": "continue",
#             "stage": 3,
#             "budget": budget,
#             "reply": "Thanks! Lastly, do you have any specific needs? (e.g., electric, low maintenance, high resale value)"
#         }

#     # Generate recommendation using existing logic
#     elif chat.stage == 3 and chat.user_input:
#         chat.needs = chat.user_input
#         recommendation = f"Based on your preferences for a {chat.car_type} under ${chat.budget} focused on '{chat.needs}', you might like:\n\n" \
#                          f"- Tesla Model 3 (great resale & EV)\n- Toyota Corolla Hybrid (affordable & efficient)\n- Kia Niro EV (family-friendly)"
        
#         return {
#             "status": "complete",
#             "reply": recommendation
#         }

#     # Default fallback
#     else:
#         return {
#             "status": "error",
#             "reply": "Sorry, I didn’t catch that. Let’s start over! What kind of car are you looking for?"
#         }

    
# @app.get("/car-specs/{model}")
# async def get_car_specs(model: str):
#     try:
#         specs = fetch_car_specs(model)
#         return JSONResponse(content={"status": "success", "specs": specs})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



from fastapi import FastAPI, HTTPException, Request, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any
from app.ai_calculations import process_car_data, fetch_car_specs, suggest_advice
from app.routes import router as routes_router

app = FastAPI(title="Car Profit and Analysis API")

# Register routes from routes.py
app.include_router(routes_router)
router = APIRouter()
# Add CORS
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


@app.get("/")
def home():
    return {"message": "Car Price Compare API is running!"}


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
            "buy_recommendations": [c.get("title") for c in processed if "BUY" in str(c.get("recommendation", "")).upper()]
        }

        return JSONResponse(content={"status": "success", "data": processed, "summary": summary})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            return JSONResponse(content={"status": "success", "data": {"car1": specs1, "car2": specs2}})

        if isinstance(payload, dict) and "cars" in payload:
            processed = process_car_data(payload["cars"])
            return JSONResponse(content={"status": "success", "data": processed})

        raise HTTPException(status_code=400, detail="Invalid payload for compare-cars. Send array of objects or {'car1':name,'car2':name}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# AI Suggestion Chatbot Route

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

@app.get("/car-specs/{model}")
async def get_car_specs(model: str):
    try:
        specs = fetch_car_specs(model)
        return JSONResponse(content={"status": "success", "specs": specs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
