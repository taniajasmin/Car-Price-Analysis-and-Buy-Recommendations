
import numpy as np
import pandas as pd
import requests
import json
import os
import re
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from difflib import SequenceMatcher

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")


def load_car_data():
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
            df = pd.read_csv(csv_path)
            print(f"Loaded cars_data.csv ({len(df)} records)")
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Failed to read cars_data.csv: {e}")

    print("No car data file found.")
    return []


# Text Extraction Helpers
def _extract_year_and_mileage_from_text(text):
    year = None
    mileage = None

    if not text:
        return year, mileage

    # Extract 4-digit year
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        year = int(m.group(0))

    # Extract mileage
    m2 = re.search(r"(\d{1,3}(?:[.,]\d{3})+|\d{4,7})\s*(km|kms|kilometer|km's)?",
                   text, flags=re.IGNORECASE)
    if m2:
        num = re.sub(r"[^\d]", "", m2.group(1))
        try:
            mileage = int(num)
        except:
            mileage = None

    return year, mileage


# Specs Lookup (fuzzy match)
def fetch_car_specs(model_str: str):
    if not model_str:
        return {"model": "Unknown"}

    model_lower = model_str.strip().lower()
    cars = load_car_data()

    if not cars:
        return {"model": model_str, "error": "No car data found."}

    best_match = None
    best_score = 0
    for car in cars:
        title = str(car.get("title", "")).lower()
        score = SequenceMatcher(None, model_lower, title).ratio()
        if score > best_score:
            best_match = car
            best_score = score

    if not best_match or best_score < 0.4:
        return {"model": model_str, "match_confidence": round(best_score, 2)}

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


# Core Calculations
def estimate_market_value(df):
    train_data = df[
        df['price_numeric'].notna() &
        df['year_numeric'].notna() &
        df['mileage_numeric'].notna()
    ].copy()

    if len(train_data) < 5:
        base_values = {'premium': 35000, 'standard': 15000}
        df['estimated_market_value'] = df.apply(
            lambda row: base_values['premium']
            if row['is_premium'] == 1
            else base_values['standard'],
            axis=1
        ) * (1 - df['age'] * 0.08)

        df.loc[df['mileage_numeric'] > 150000, 'estimated_market_value'] *= 0.8
        df.loc[df['mileage_numeric'] > 200000, 'estimated_market_value'] *= 0.7

    else:
        features = ['year_numeric', 'mileage_numeric', 'is_premium']
        X = train_data[features].fillna(0)
        y = train_data['price_numeric']

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=10
        )
        model.fit(X, y)

        X_all = df[features].fillna(0)
        df['estimated_market_value'] = model.predict(X_all)

    return df


def calculate_costs_and_profit(df):
    def is_ev(row):
        title_lower = str(row['title']).lower()
        brand_lower = str(row['brand']).lower()
        ev_keywords = ['electric', 'ev']
        ev_brands = ['tesla', 'rivian', 'polestar']
        return any(k in title_lower for k in ev_keywords) or brand_lower in ev_brands

    df['is_ev'] = df.apply(is_ev, axis=1)

    df['inspection_cost'] = 150
    df['transport_cost'] = 200
    df['repair_cost'] = 0

    # Age-based costs
    df.loc[(df['age'] > 5) & (~df['is_ev']), 'repair_cost'] += 400
    df.loc[(df['age'] > 10) & (~df['is_ev']), 'repair_cost'] += 800
    df.loc[(df['age'] > 15) & (~df['is_ev']), 'repair_cost'] += 1200
    df.loc[(df['age'] > 20) & (~df['is_ev']), 'repair_cost'] += 2000

    df.loc[(df['age'] > 5) & df['is_ev'], 'repair_cost'] += 200
    df.loc[(df['age'] > 10) & df['is_ev'], 'repair_cost'] += 400
    df.loc[(df['age'] > 15) & df['is_ev'], 'repair_cost'] += 600
    df.loc[(df['age'] > 20) & df['is_ev'], 'repair_cost'] += 1000

    # Mileage-based costs
    df.loc[(df['mileage_numeric'] > 80000) & (~df['is_ev']), 'repair_cost'] += 300
    df.loc[(df['mileage_numeric'] > 150000) & (~df['is_ev']), 'repair_cost'] += 500
    df.loc[(df['mileage_numeric'] > 200000) & (~df['is_ev']), 'repair_cost'] += 1000
    df.loc[(df['mileage_numeric'] > 250000) & (~df['is_ev']), 'repair_cost'] += 1500

    df.loc[(df['mileage_numeric'] > 100000) & df['is_ev'], 'repair_cost'] += 150
    df.loc[(df['mileage_numeric'] > 160000) & df['is_ev'], 'repair_cost'] += 300
    df.loc[(df['mileage_numeric'] > 250000) & df['is_ev'], 'repair_cost'] += 700
    df.loc[(df['mileage_numeric'] > 300000) & df['is_ev'], 'repair_cost'] += 2000

    df['total_costs'] = (
        df['inspection_cost'] +
        df['transport_cost'] +
        df['repair_cost']
    )

    df['expected_profit'] = (
        df['estimated_market_value'] -
        df['price_numeric'] -
        df['total_costs']
    )

    df['profit_margin_pct'] = (
        df['expected_profit'] /
        df['price_numeric'].replace(0, float('nan'))
    ) * 100

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

    df['risk_level'] = "Low"
    df.loc[df['risk_score'] > 3, 'risk_level'] = "Medium"
    df.loc[df['risk_score'] > 6, 'risk_level'] = "High"

    df['recommendation'] = "DON'T BUY"
    df.loc[
        (df['expected_profit'] > 2000) &
        (df['profit_margin_pct'] > 20) &
        (df['risk_level'] == 'Low'),
        'recommendation'
    ] = "BUY"

    df.loc[
        (df['expected_profit'] > 5000) &
        (df['profit_margin_pct'] > 50) &
        (df['risk_level'] == 'Low'),
        'recommendation'
    ] = "STRONG BUY"

    return df


# Main Processing Pipeline
def process_car_data(car_data_list):
    rows = []
    for car in car_data_list:
        try:
            row = car.dict() if hasattr(car, "dict") else dict(car)
        except Exception:
            row = dict(car) if isinstance(car, dict) else {}
        rows.append(row)

    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Normalize price -> price_numeric
    if 'price_numeric' not in df.columns:
        if 'price' in df.columns:
            df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce')
        else:
            df['price_numeric'] = None

    # Fix abnormal prices
    df.loc[df['price_numeric'] < 50, 'price_numeric'] *= 1000
    df.loc[df['price_numeric'] > 500000, 'price_numeric'] = None

    # Extract year/mileage from raw_text
    if 'raw_text' in df.columns:
        extracted_years = []
        extracted_mileage = []

        for t in df['raw_text'].fillna(''):
            y, m = _extract_year_and_mileage_from_text(t)
            extracted_years.append(y)
            extracted_mileage.append(m)

        if 'year_numeric' not in df.columns:
            df['year_numeric'] = extracted_years
        else:
            df['year_numeric'] = df['year_numeric'].fillna(pd.Series(extracted_years))

        if 'mileage_numeric' not in df.columns:
            df['mileage_numeric'] = extracted_mileage
        else:
            df['mileage_numeric'] = df['mileage_numeric'].fillna(pd.Series(extracted_mileage))

    # Fill missing brand
    if 'brand' not in df.columns:
        df['brand'] = df['title'].str.split().str[0].str.capitalize()

    # Calculate age from year
    current_year = pd.Timestamp.now().year
    df['age'] = df.apply(
        lambda r: current_year - r['year_numeric'] if pd.notna(r['year_numeric']) else None,
        axis=1
    )
    df['age'] = df['age'].fillna(5)

    # Determine premium brand
    premium_brands = ['bmw', 'audi', 'mercedes', 'porsche', 'tesla', 'lexus', 'volvo']
    df['is_premium'] = df['brand'].str.lower().isin(premium_brands).astype(int)

    # Fill missing numeric columns safely
    for col in ['year_numeric', 'mileage_numeric', 'age', 'is_premium']:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].fillna(0)

    # Main pipeline
    df = estimate_market_value(df)
    df = calculate_costs_and_profit(df)
    df = assess_risk(df)

    # Clean output (convert numpy types)
    output = []
    for record in df.to_dict(orient='records'):
        clean = {}
        for k, v in record.items():
            if isinstance(v, (np.integer, np.floating)):
                clean[k] = v.item()
            else:
                clean[k] = v
        output.append(clean)

    return output
