import os
import json
import re
from difflib import SequenceMatcher
from typing import List, Any, Dict

import numpy as np
import pandas as pd

from openai import OpenAI

# SAFE CLIENT CREATION
def get_client():
    """Return a fresh OpenAI client every time (prevents reload errors)."""
    return OpenAI()  # reads OPENAI_API_KEY automatically


# —————————————————————————
# Brand groups
# —————————————————————————
PREMIUM_BRANDS = {"bmw", "audi", "mercedes", "porsche", "tesla", "lexus", "volvo"}
RELIABLE_BRANDS = {"toyota", "honda", "mazda", "nissan"}
UNRELIABLE_BRANDS = {"fiat", "peugeot", "renault", "land rover"}


# —————————————————————————
# Load scraped car dataset
# —————————————————————————
def load_car_data():
    json_path = os.path.join(os.getcwd(), "cars_data.json")
    csv_path = os.path.join(os.getcwd(), "cars_data.csv")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df.to_dict(orient="records")
        except:
            pass

    return []


# —————————————————————————
# Text extraction
# —————————————————————————
def _extract_year_mileage(text: str):
    year = None
    mileage = None
    if not text:
        return year, mileage

    # Year
    m_year = re.search(r"\b(19|20)\d{2}\b", text)
    if m_year:
        year = int(m_year.group(0))

    # Mileage
    m_km = re.search(
        r"(\d{1,3}(?:[.,]\d{3})+|\d{4,7})\s*(km|kms|kilometer)?",
        text,
        flags=re.IGNORECASE
    )
    if m_km:
        digits = re.sub(r"[^\d]", "", m_km.group(1))
        try:
            mileage = int(digits)
        except:
            mileage = None

    return year, mileage


def _is_ev(title: str, brand: str):
    t = (title or "").lower()
    b = (brand or "").lower()
    if "ev" in t or "electric" in t or "phev" in t:
        return True
    if b in {"tesla", "polestar", "rivian"}:
        return True
    return False


# —————————————————————————
# GPT-assisted value (hidden)
# —————————————————————————
def _gpt_value(title, brand, year, mileage):
    """GPT valuation with safe client initialization."""
    try:
        client = get_client()

        prompt = (
            "Estimate realistic used car value. "
            "Return ONLY a number.\n"
            f"Title: {title}\nBrand: {brand}\nYear: {year}\nMileage: {mileage}\n"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )

        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"[^\d]", "", raw)

        return float(raw) if raw else None
    except:
        return None


# —————————————————————————
# Market value estimation
# —————————————————————————
def _dataset_baseline(brand):
    cars = load_car_data()
    prices_all = []
    prices_brand = []
    brand = brand.lower()

    for c in cars:
        p = c.get("price_numeric") or c.get("price")
        try:
            p = float(p)
        except:
            continue

        if 1000 < p < 300000:
            prices_all.append(p)

            if str(c.get("brand", "")).lower() == brand:
                prices_brand.append(p)

    if len(prices_brand) >= 4:
        return float(pd.Series(prices_brand).median())

    if prices_all:
        return float(pd.Series(prices_all).median())

    return 20000.0


def _estimate_value(row):
    brand = str(row.get("brand", "")).lower()
    year = int(row.get("year_numeric") or 0)
    mileage = int(row.get("mileage_numeric") or 0)
    title = row.get("title", "")
    age = int(row.get("age") or 5)

    base = _dataset_baseline(brand)

    if brand in PREMIUM_BRANDS:
        base *= 1.2
    elif brand in RELIABLE_BRANDS:
        base *= 1.1
    elif brand in UNRELIABLE_BRANDS:
        base *= 0.9

    if age <= 2:
        base *= 1.05
    elif age <= 5:
        base *= 1
    elif age <= 10:
        base *= 0.9
    else:
        base *= 0.75

    if mileage < 60000:
        base *= 1.05
    elif mileage < 100000:
        base *= 1
    elif mileage < 150000:
        base *= 0.9
    else:
        base *= 0.8

    if _is_ev(title, brand):
        base *= 1.1

    gpt_val = None
    if year >= 2018:
        gpt_val = _gpt_value(title, brand, year, mileage)

    if gpt_val and 3000 < gpt_val < 200000:
        base = (base * 0.6) + (gpt_val * 0.4)

    lower = max(5000, base * 0.6)
    upper = base * 1.8 + 8000
    return float(max(lower, min(base, upper)))


# —————————————————————————
# Costs / Risk / Recommendation
# —————————————————————————
def _estimate_costs(row):
    mileage = row.get("mileage_numeric", 100000)
    base = 350

    if mileage < 80000:
        repair = 500
    elif mileage < 150000:
        repair = 1000
    else:
        repair = 1800

    return float(base + repair)


def _risk(row):
    score = 0
    age = row.get("age", 5)
    mileage = row.get("mileage_numeric", 100000)
    brand = row.get("brand", "").lower()

    if age > 15:
        score += 4
    elif age > 10:
        score += 3
    elif age > 7:
        score += 2
    elif age > 3:
        score += 1

    if mileage > 200000:
        score += 4
    elif mileage > 150000:
        score += 3
    elif mileage > 100000:
        score += 2
    elif mileage > 60000:
        score += 1

    if brand in UNRELIABLE_BRANDS:
        score += 2
    elif brand in RELIABLE_BRANDS:
        score -= 1

    return max(score, 0)


def _risk_level(score):
    if score <= 2:
        return "Low"
    if score <= 5:
        return "Medium"
    return "High"


def _recommendation(profit, margin, risk):
    if profit > 4000 and margin > 0.2 and risk <= 2:
        return "STRONG BUY"
    if profit > 2500 and margin > 0.12 and risk <= 4:
        return "BUY"
    if profit > 1000 and margin > 0.05 and risk <= 5:
        return "CONSIDER"
    return "DON'T BUY"


# —————————————————————————
# Specs lookup
# —————————————————————————
def fetch_car_specs(model):
    if not model:
        return {"model": "Unknown"}

    cars = load_car_data()
    if not cars:
        return {"model": model, "error": "No data"}

    q = model.lower()
    best = None
    best_score = 0

    for c in cars:
        title = str(c.get("title", "")).lower()
        if " in " in title:
            continue

        score = SequenceMatcher(None, q, title).ratio()
        if q in title:
            score += 0.15

        if score > best_score:
            best_score = score
            best = c

    if not best or best_score < 0.5:
        return {"model": model, "match_confidence": best_score}

    return {
        "model": best.get("title"),
        "brand": best.get("brand"),
        "price_numeric": best.get("price_numeric"),
        "year_numeric": best.get("year_numeric"),
        "mileage_numeric": best.get("mileage_numeric"),
        "age": best.get("age"),
        "is_premium": best.get("is_premium"),
        "url": best.get("url"),
        "match_confidence": best_score,
    }


# —————————————————————————
# Main pipeline
# —————————————————————————
def process_car_data(cars_list: List[Any]):
    rows = []
    for c in cars_list:
        try:
            rows.append(c.dict() if hasattr(c, "dict") else dict(c))
        except:
            rows.append(dict(c) if isinstance(c, dict) else {})

    if not rows:
        return []

    df = pd.DataFrame(rows)

    if "title" not in df.columns:
        df["title"] = "Unknown"

    if "brand" not in df.columns:
        df["brand"] = df["title"].str.split().str[0].str.capitalize()

    df["brand"] = df["brand"].fillna(
        df["title"].str.split().str[0].str.capitalize()
    ).astype(str)

    if "price_numeric" not in df.columns:
        if "price" in df.columns:
            df["price_numeric"] = pd.to_numeric(df["price"], errors="coerce")
        else:
            df["price_numeric"] = 0

    df["price_numeric"] = pd.to_numeric(df["price_numeric"], errors="coerce").fillna(0)
    df.loc[df["price_numeric"] < 50, "price_numeric"] *= 1000

    if "raw_text" in df.columns:
        years = []
        miles = []
        for t in df["raw_text"].fillna(""):
            y, m = _extract_year_mileage(t)
            years.append(y)
            miles.append(m)

        df["year_numeric"] = df.get("year_numeric", pd.Series(years)).fillna(pd.Series(years))
        df["mileage_numeric"] = df.get("mileage_numeric", pd.Series(miles)).fillna(pd.Series(miles))

    df["year_numeric"] = pd.to_numeric(df["year_numeric"], errors="coerce").fillna(0)
    df["mileage_numeric"] = pd.to_numeric(df["mileage_numeric"], errors="coerce").fillna(0)

    current_year = pd.Timestamp.now().year

    def calc_age(r):
        y = int(r["year_numeric"])
        if 1900 < y <= current_year:
            return current_year - y
        return 5

    df["age"] = df.apply(calc_age, axis=1)

    results = []

    for _, row in df.iterrows():
        r = dict(row)

        value = _estimate_value(r)
        costs = _estimate_costs(r)

        price = float(r.get("price_numeric") or 0)
        if price <= 0:
            price = value * 0.85

        profit = value - (price + costs)
        margin = profit / price if price > 0 else 0

        risk = _risk(r)
        rlevel = _risk_level(risk)
        rec = _recommendation(profit, margin, risk)

        r["estimated_market_value"] = round(value, 2)
        r["total_costs"] = round(costs, 2)
        r["price_numeric"] = round(price, 2)
        r["expected_profit"] = round(profit, 2)
        r["profit_margin_pct"] = round(margin * 100, 2)
        r["risk_score"] = risk
        r["risk_level"] = rlevel
        r["recommendation"] = rec

        results.append(r)

    cleaned = []
    for r in results:
        out = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.floating)):
                out[k] = v.item()
            else:
                out[k] = v
        cleaned.append(out)

    return cleaned
