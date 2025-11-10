import numpy as np
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
import openai
import re

from sklearn.ensemble import RandomForestRegressor
from difflib import SequenceMatcher

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")  

def load_car_data():
    """
    Try to load from cars_data.json (preferred) or fallback to cars_data.csv.
    """
    json_path = os.path.join(os.getcwd(), "cars_data.json")
    csv_path = os.path.join(os.getcwd(), "cars_data.csv")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"Loaded cars_data.json ({len(data)} records)")
                return data
        except Exception as e:
            print(f"Failed to read cars_data.json: {e}")

    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"Loaded cars_data.csv ({len(df)} records)")
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Failed to read cars_data.csv: {e}")

    print("No car data file found.")
    return []


def fetch_car_specs(model_str):
    """
    Safely find the closest matching car from cars_data.csv.
    Returns available specs (brand, year, mileage, price, etc.)
    """
    if not model_str:
        return {
            "model": model_str or "Unknown",
            "brand": "N/A",
            "year": "N/A",
            "mileage": "N/A",
            "price": None
        }

    model_lower = model_str.strip().lower()
    cars = load_car_data()
    if not cars:
        return {
            "model": model_str,
            "brand": "N/A",
            "year": "N/A",
            "mileage": "N/A",
            "price": None
        }

    # Direct substring match
    for car in cars:
        title = str(car.get("title", "")).lower()
        if model_lower in title:
            return {
                "model": car.get("title", "Unknown"),
                "brand": car.get("brand", "Unknown"),
                "year": car.get("year_numeric", "N/A"),
                "mileage": car.get("mileage_numeric", "N/A"),
                "price": car.get("price_numeric", None),
            }

    # Fuzzy match if no direct hit
    best_match = None
    best_score = 0
    for car in cars:
        title = str(car.get("title", "")).lower()
        score = SequenceMatcher(None, model_lower, title).ratio()
        if score > best_score:
            best_match = car
            best_score = score

    if best_match and best_score > 0.3:  # avoid random matches
        return {
            "model": best_match.get("title", "Unknown"),
            "brand": best_match.get("brand", "Unknown"),
            "year": best_match.get("year_numeric", "N/A"),
            "mileage": best_match.get("mileage_numeric", "N/A"),
            "price": best_match.get("price_numeric", None),
            "match_confidence": round(best_score, 2)
        }

    # Graceful fallback if nothing close enough
    print(f"No match found for '{model_str}'")
    return {
        "model": model_str,
        "brand": "N/A",
        "year": "N/A",
        "mileage": "N/A",
        "price": None
    }


def estimate_market_value(df):
    train_data = df[df['price_numeric'].notna() & df['year_numeric'].notna() & df['mileage_numeric'].notna()].copy()
    if len(train_data) < 5:
        base_values = {'premium': 35000, 'standard': 15000}
        df['estimated_market_value'] = df.apply(lambda row: base_values['premium'] if row['is_premium'] == 1 else base_values['standard'], axis=1) * (1 - df['age'] * 0.08)
        df.loc[df['mileage_numeric'] > 150000, 'estimated_market_value'] *= 0.8
        df.loc[df['mileage_numeric'] > 200000, 'estimated_market_value'] *= 0.7
    else:
        features = ['year_numeric', 'mileage_numeric', 'is_premium']
        X = train_data[features].fillna(0)
        y = train_data['price_numeric']
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X, y)
        X_all = df[features].fillna(0)
        df['estimated_market_value'] = model.predict(X_all)
    return df

def calculate_costs_and_profit(df):
    def is_ev(row):
        title_lower = str(row['title']).lower() if pd.notna(row['title']) else ''
        brand_lower = str(row['brand']).lower() if pd.notna(row['brand']) else ''
        ev_keywords = ['electric', 'ev']
        ev_brands = ['tesla', 'rivian', 'polestar']
        return any(kw in title_lower for kw in ev_keywords) or brand_lower in ev_brands
    
    df['is_ev'] = df.apply(is_ev, axis=1)
    
    df['inspection_cost'] = 150
    df['transport_cost'] = 200
    df['repair_cost'] = 0
    
    # Age-based repairs
    df.loc[(df['age'] > 5) & (~df['is_ev']), 'repair_cost'] += 400
    df.loc[(df['age'] > 10) & (~df['is_ev']), 'repair_cost'] += 800
    df.loc[(df['age'] > 15) & (~df['is_ev']), 'repair_cost'] += 1200
    df.loc[(df['age'] > 20) & (~df['is_ev']), 'repair_cost'] += 2000
    df.loc[(df['age'] > 5) & df['is_ev'], 'repair_cost'] += 200
    df.loc[(df['age'] > 10) & df['is_ev'], 'repair_cost'] += 400
    df.loc[(df['age'] > 15) & df['is_ev'], 'repair_cost'] += 600
    df.loc[(df['age'] > 20) & df['is_ev'], 'repair_cost'] += 1000
    
    # Mileage-based repairs
    df.loc[(df['mileage_numeric'] > 80000) & (~df['is_ev']), 'repair_cost'] += 300
    df.loc[(df['mileage_numeric'] > 150000) & (~df['is_ev']), 'repair_cost'] += 500
    df.loc[(df['mileage_numeric'] > 200000) & (~df['is_ev']), 'repair_cost'] += 1000
    df.loc[(df['mileage_numeric'] > 250000) & (~df['is_ev']), 'repair_cost'] += 1500
    df.loc[(df['mileage_numeric'] > 100000) & df['is_ev'], 'repair_cost'] += 150
    df.loc[(df['mileage_numeric'] > 160000) & df['is_ev'], 'repair_cost'] += 300
    df.loc[(df['mileage_numeric'] > 250000) & df['is_ev'], 'repair_cost'] += 700
    df.loc[(df['mileage_numeric'] > 300000) & df['is_ev'], 'repair_cost'] += 2000
    
    df['total_costs'] = df['inspection_cost'] + df['transport_cost'] + df['repair_cost']
    df['expected_profit'] = df['estimated_market_value'] - df['price_numeric'] - df['total_costs']
    df['profit_margin_pct'] = (df['expected_profit'] / df['price_numeric'].replace(0, float('nan'))) * 100
    
    return df

def assess_risk(df):
    df['risk_score'] = 0
    df.loc[df['age'] > 10, 'risk_score'] += 2
    df.loc[df['age'] > 15, 'risk_score'] += 3
    df.loc[df['age'] > 20, 'risk_score'] += 5
    df.loc[df['mileage_numeric'] > 150000, 'risk_score'] += 2
    df.loc[df['mileage_numeric'] > 200000, 'risk_score'] += 4
    df.loc[df['mileage_numeric'] > 250000, 'risk_score'] += 6
    df.loc[df['is_premium'] == 1, 'risk_score'] -= 1
    df['risk_score'] = df['risk_score'].clip(lower=0)
    df['risk_level'] = 'Low'
    df.loc[df['risk_score'] > 3, 'risk_level'] = 'Medium'
    df.loc[df['risk_score'] > 6, 'risk_level'] = 'High'
    
    # Recommendation logic
    df['recommendation'] = "DON'T BUY"
    df.loc[(df['expected_profit'] > 2000) & (df['profit_margin_pct'] > 20) & (df['risk_level'] == 'Low'), 'recommendation'] = 'BUY'
    df.loc[(df['expected_profit'] > 5000) & (df['profit_margin_pct'] > 50) & (df['risk_level'] == 'Low'), 'recommendation'] = 'STRONG BUY'
    
    return df

def fetch_car_specs(model_str: str):
    """
    Find a car in cars_data.json/csv that best matches the given model name.
    Return only real scraped fields.
    """
    if not model_str:
        return {"model": "Unknown"}

    model_lower = model_str.strip().lower()
    cars = load_car_data()

    if not cars:
        print("No car data loaded.")
        return {"model": model_str, "error": "No car data found."}

    # Fuzzy match titles
    best_match = None
    best_score = 0
    for car in cars:
        title = str(car.get("title", "")).lower()
        score = SequenceMatcher(None, model_lower, title).ratio()
        if score > best_score:
            best_match = car
            best_score = score

    if not best_match or best_score < 0.4:
        print(f"No close match found for '{model_str}'")
        return {"model": model_str, "match_confidence": round(best_score, 2)}

    # Return only keys that actually exist in your scraped data
    return {
        "model": best_match.get("title", model_str),
        "brand": best_match.get("brand", "Unknown"),
        "price_numeric": best_match.get("price_numeric", None),
        "year_numeric": best_match.get("year_numeric", None),
        "mileage_numeric": best_match.get("mileage_numeric", None),
        "age": best_match.get("age", None),
        "is_premium": best_match.get("is_premium", None),
        "url": best_match.get("url", None),
        "match_confidence": round(best_score, 2)
    }

def process_car_data(car_data_list):
    """
    Accepts a list of either Pydantic models or plain dicts.
    Returns a list of processed car dicts with calculated fields.
    """
    rows = []
    for car in car_data_list:
        # If it's a Pydantic model, call .dict(), otherwise assume dict-like
        try:
            row = car.dict() if hasattr(car, "dict") else dict(car)
        except Exception:
            # Fallback: try to convert to dict safely
            row = dict(car) if isinstance(car, dict) else {}
        rows.append(row)

    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Ensure numeric columns exist so downstream logic won't crash
    for col in ['price_numeric', 'year_numeric', 'mileage_numeric', 'is_premium', 'age']:
        if col not in df.columns:
            df[col] = 0

    # run calculations
    df = estimate_market_value(df)
    df = calculate_costs_and_profit(df)
    df = assess_risk(df)

    # Convert numpy types to native Python types for JSON safety
    result = df.to_dict(orient='records')
    normalized = []
    for r in result:
        r2 = {}
        for k, v in r.items():
            # convert numpy types
            if isinstance(v, (np.integer, np.floating)):
                v = v.item()
            r2[k] = v
        normalized.append(r2)
    return normalized


# def suggest_advice(message, budget=None, needs=None):
#     try:
#         client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         prompt = f"User query: {message}. Budget: {budget or 'not specified'}. Needs: {needs or 'not specified'}. Provide professional car buying advice."
#         response = client.chat.completions.create(
#             model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),  
#             messages=[
#                 {"role": "system", "content": "You are a professional car advisor with expertise in family vehicles, budgets, and market trends. Respond naturally, warmly, and helpfully."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=150, 
#             temperature=0.7  
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return "I'm having a brief connection issue with my advice system. Based on general knowledge: The Tesla Model 3 2025 is efficient for families but check space for car seats. Let's try again soon!"
