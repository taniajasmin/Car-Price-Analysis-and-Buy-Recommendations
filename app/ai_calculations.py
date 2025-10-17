from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
import openai


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")  # Default to gpt-4 if not set

# Load CSV data
def load_car_data():
    try:
        df = pd.read_csv('cars_data.csv')
        return df.to_dict(orient='records')
    except FileNotFoundError:
        print("cars_data.csv not found. Please run the scraper first.")
        return []
    
# def scrape_with_thordata(url):
#     headers = {"Authorization": f"Bearer {os.getenv('THORDATA_API_KEY')}", "Content-Type": "application/x-www-form-urlencoded"}
#     payload = {"url": url, "type": "html", "js_render": "True"}
#     try:
#         response = requests.post("https://universalapi.thordata.com/request", data=payload, headers=headers)
#         if response.status_code == 200:
#             data = response.json()
#             return BeautifulSoup(data.get('html', ''), 'html.parser')
#         else:
#             print(f"Scraping error: Status {response.status_code}")
#             return None
#     except Exception as e:
#         print(f"Scraping failed: {str(e)}")
#         return None

# def scrape_car_listings():
#     urls = ["https://www.2dehands.be/", "https://www.autoscout24.com/"]  # Base URLs, adjust as needed
#     all_cars = []
#     for url in urls:
#         soup = scrape_with_thordata(url)
#         if soup:
#             # Simplified: Extract listings (mimic your Colab logic)
#             listings = soup.find_all('div', class_='listing')  # Adjust class based on site
#             for listing in listings:
#                 title = listing.find('h2', class_='title').text.strip() if listing.find('h2', class_='title') else "N/A"
#                 price = listing.find('span', class_='price').text.replace('€', '').replace('.', '').strip() if listing.find('span', class_='price') else None
#                 url = listing.find('a')['href'] if listing.find('a') else url
#                 all_cars.append({"title": title, "url": url, "price_numeric": float(price) if price else None})
#     return all_cars

# def schedule_scraping():
#     schedule.every(15).days.do(lambda: process_and_store_scraped_data())
#     while True:
#         schedule.run_pending()
#         time.sleep(60)  # Check every minute

# def process_and_store_scraped_data():
#     cars = scrape_car_listings()
#     # Save to json file 
#     with open('scraped_cars.json', 'w') as f:
#         json.dump(cars, f)
#     print(f"Scraped {len(cars)} cars at {time.ctime()}")


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

def fetch_car_specs(model_str):
    cars = load_car_data()
    for car in cars:
        if model_str.lower() in car['title'].lower():
            return {
                'engine': 'N/A',  # Placeholder; add if scraped
                'fuel_type': 'N/A',  # Placeholder; add if scraped
                'max_power': 'N/A',  # Placeholder; add if scraped
                'driving_range': 'N/A',  # Placeholder; add if scraped
                'drivetrain': 'N/A',  # Placeholder; add if scraped
                'price': car.get('price_numeric')
            }
    print(f"No matching specs found for {model_str}")
    return {'engine': 'N/A', 'fuel_type': 'N/A', 'max_power': 'N/A', 'driving_range': 'N/A', 'drivetrain': 'N/A', 'price': None}

def process_car_data(car_data_list):
    df = pd.DataFrame([car.dict() for car in car_data_list])
    df = estimate_market_value(df)
    df = calculate_costs_and_profit(df)
    df = assess_risk(df)
    return df.to_dict(orient='records')

# def fetch_car_specs(model_str):
#     try:
#         parts = model_str.split()
#         make = parts[0]
#         year = int(parts[-1]) if parts[-1].isdigit() else 2025
#         model = ' '.join(parts[1:-1]) if parts[-1].isdigit() else ' '.join(parts[1:])
#         url = f"https://www.carqueryapi.com/api/0.3/?cmd=getTrims&make={make}&year={year}&model={model}"
#         response = requests.get(url, timeout=10)
#         if response.status_code == 200:
#             data = json.loads(response.text.replace('jqcep(', '').replace(');', ''))
#             trims = data.get('Trims', [])
#             if trims:
#                 trim = trims[0]
#                 return {
#                     'engine': trim.get('model_engine_type', 'N/A'),
#                     'fuel_type': trim.get('model_engine_fuel', 'N/A'),
#                     'max_power': f"{trim.get('model_engine_power_ps', 'N/A')} hp",
#                     'driving_range': f"{trim.get('model_range_km', 'N/A')} km",
#                     'drivetrain': trim.get('model_drive', 'N/A')
#                 }
#         print(f"Failed to fetch specs for {model_str}")
#         return {'engine': 'N/A', 'fuel_type': 'N/A', 'max_power': 'N/A', 'driving_range': 'N/A', 'drivetrain': 'N/A'}
#     except Exception as e:
#         print(f"Error fetching specs for {model_str}: {str(e)}")
#         return {'engine': 'N/A', 'fuel_type': 'N/A', 'max_power': 'N/A', 'driving_range': 'N/A', 'drivetrain': 'N/A'}
    
# def suggest_advice(message, budget=None, needs=None):
#     # Simple rule-based chat response
#     if "budget" in message.lower():
#         return f"Good Morning! With a ${budget or 20000} budget, I recommend family-friendly options like Toyota Camry. Great options!"
#     elif "meet" in message.lower():
#         return "We can meet tomorrow."
#     elif "thanks" in message.lower():
#         return "That will be great! Glad to hear you're free."
#     else:
#         return "Okay, thanks for your time."


def suggest_advice(message, budget=None, needs=None):
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"User query: {message}. Budget: {budget or 'not specified'}. Needs: {needs or 'not specified'}. Provide professional car buying advice."
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),  
            messages=[
                {"role": "system", "content": "You are a professional car advisor with expertise in family vehicles, budgets, and market trends. Respond naturally, warmly, and helpfully."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150, 
            temperature=0.7  
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return "I'm having a brief connection issue with my advice system. Based on general knowledge: The Tesla Model 3 2025 is efficient for families but check space for car seats. Let's try again soon!"