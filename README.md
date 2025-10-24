# Car Price Analysis & AI Recommendations

This project analyzes car listings, compares them, and gives AI-powered buy/sell recommendations.
It uses the ThorData API for scraping real-world listings and applies custom AI math logic to calculate market value, profit, and risk.

---

## Overview

- Scraper to ThorData (JS-rendered pages) to BeautifulSoup to `cars_data.csv`
- Analysis to Market-value, depreciation, profit-risk scores
- AI to OpenAI (`gpt-4o-mini` by default) for personalized recommendations
- API to FastAPI (Uvicorn)

---

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

## 2. Create .env
```bash
THORDATA_API_KEY=your_thordata_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini   # optional
```

## 3. run Locally
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```


## API Endpoints & cURL Examples
### 1. /analyze-cars/ – Analyze listings
```bash
curl -X POST http://127.0.0.1:8000/analyze-cars/ \
  -H "Content-Type: application/json" \
  -d '[{"title":"Tesla Model 3 2025","brand":"Tesla","year_numeric":2025,"mileage_numeric":12000,"price_numeric":42000,"is_premium":1,"age":0}]'
```

### 2. /compare-cars/ – Compare two cars
By objects
```bash
curl -X POST http://127.0.0.1:8000/compare-cars/ \
  -H "Content-Type: application/json" \
  -d '[{"title":"Tesla Model 3 2025","price_numeric":42000},{"title":"BMW M3 2023","price_numeric":58000}]'
  ```
By names
```bash
curl -X POST http://127.0.0.1:8000/compare-cars/ \
  -H "Content-Type: application/json" \
  -d '{"car1":"Tesla Model 3 2025","car2":"BMW 1974"}'
```

### 3. /ai-suggest/ – AI Recommendations
```bash
curl -X POST http://127.0.0.1:8000/ai-suggest/ \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Suggest an EV under 40k with good resale","budget":40000,"needs":"family"}'
```

## ThorData Scraper (scraper.py)

- Calls: https://universalapi.thordata.com/request
- Returns: Fully rendered HTML (JS support)
- Parses with: BeautifulSoup
- Extracts: title, price, year, mileage, brand, age, is_premium
- Saves to: cars_data.csv


Run manually:
```bash
python -c "from app.scraper import scrape_and_save; scrape_and_save()"
```

## Project Structure

```text
Car-Price-Analysis-and-Buy-Recommendations/
├── app/
│   ├── main.py
│   ├── ai_calculations.py
│   └── scraper.py
├── cars_data.csv          # auto-generated
├── requirements.txt
├── .env                   # create this
├── .gitignore
└── README.txt             # this file
```

## Security & Production

| Item         | Recommendation                          |
|--------------|-----------------------------------------|
| `.env`       | Add to `.gitignore`                     |
| Secrets      | Use Vault / AWS Secrets Manager / cloud env vars |
| Rate Limits  | Implement retry + exponential backoff   |
| Logging      | Use `loguru` or `structlog`             |
| Validation   | Pydantic models enforce input           |
