# Car Price Analysis & Buy Recommendation

**Scrape real-time car listings, analyze market value, calculate profit/risk, and get AI-powered buy suggestions.**

This FastAPI backend uses **ThorData** for web scraping and **OpenAI** for intelligent recommendations. Compare cars, detect deals, and make data-driven purchase decisions.

---

## Features

- **Real-time scraping** via ThorData (JS-rendered pages supported)
- **Market value analysis** with profit/risk scoring
- **Smart car comparison** (by name or full spec)
- **AI-powered suggestions** using GPT (`gpt-4o-mini`)
- **CSV export** of scraped listings
- **Production-ready** with `.env`, logging, and error handling

---

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
``` 

## 2. Create .env (important)

Create a file named .env in the project root (car_price_compare - Copy/.env) with the following variables:
```bash
# .env (place at project root)
THORDATA_API_KEY=your_thordata_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini   # optional, default used if missing
```

**Notes:**

- THORDATA_API_KEY is required for scraping with ThorData.

- OPENAI_API_KEY is required only if you use the AI suggestion endpoint.

- The project uses python-dotenv (load_dotenv()), so variables in .env are loaded automatically.

- Do NOT commit .env to git. Add .env to .gitignore.

## 3. Run backend
```bash
uvicorn app.main:app --reload
```

## Endpoints & cURL Examples
Analyze Cars
```bash
curl -X POST http://127.0.0.1:8000/analyze-cars/ \
-H "Content-Type: application/json" \
-d '[{"title":"Tesla Model 3 2025","brand":"Tesla","year_numeric":2025,"mileage_numeric":12000,"price_numeric":42000,"is_premium":1,"age":0}]'
 ```
Compare Cars (by names or objects)

Array mode:
```bash
curl -X POST http://127.0.0.1:8000/compare-cars/ \
-H "Content-Type: application/json" \
-d '[{"title":"Tesla Model 3 2025","price_numeric":42000}]'
```

Name mode:
```bash
curl -X POST http://127.0.0.1:8000/compare-cars/ \
-H "Content-Type: application/json" \
-d '{"car1":"Tesla Model 3 2025","car2":"BMW 1974"}'
```

## AI Suggest
```bash
curl -X POST http://127.0.0.1:8000/ai-suggest/ \
-H "Content-Type: application/json" \
-d '{"prompt":"Suggest an EV under 40k with good resale","budget":40000,"needs":"family"}'
```

## ThorData Scraper (how it works)

- scraper.py calls https://universalapi.thordata.com/request with THORDATA_API_KEY.

- ThorData returns rendered HTML (supports JavaScript).

- BeautifulSoup parses listings and parse_car_data() extracts title, price_numeric, year_numeric, mileage_numeric, brand, age, is_premium.

- Parsed results are saved to cars_data.json.

🧾 Project structure
```text
Car-Price-Analysis-and-Buy-Recommendations/
├── app/
│   ├── main.py
│   ├── ai_calculations.py
│   └── scraper.py
├── cars_data.csv
├── requirements.txt
├── .env                
└── README.md
```


## Security & Deployment Notes

- Add .env to .gitignore.

- Use secrets manager (Vault/Cloud Secrets) for production — do not keep keys in repo.

- Rate limits: ThorData and OpenAI have quotas — handle throttling/retries in production.
