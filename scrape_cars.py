import http.client
import re
import json
import time
import schedule
from datetime import datetime
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv 

load_dotenv()

THORDATA_API_KEY = os.getenv("THORDATA_API_KEY")
if not THORDATA_API_KEY:
    raise ValueError("THORDATA_API_KEY not found in .env file!")

print(f"Loaded API key: {THORDATA_API_KEY[:10]}...{THORDATA_API_KEY[-4:]}")

# THORDATA (http.client)
def scrape_with_thordata(url: str) -> BeautifulSoup | None:
    conn = http.client.HTTPSConnection("universalapi.thordata.com")
    payload = {"url": url, "type": "html", "js_render": "True"}
    form_data = urlencode(payload)
    headers = {
        'Authorization': f'Bearer {THORDATA_API_KEY}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    try:
        conn.request("POST", "/request", form_data, headers)
        res = conn.getresponse()
        if res.status == 200:
            data = json.loads(res.read().decode("utf-8"))
            html = data.get("html", "")
            if html.strip():
                dbg = f"debug_thordata_{url.split('//')[1].split('/')[0]}.html"
                with open(dbg, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"[Thordata] Saved {dbg} ({len(html):,} chars)")
                return BeautifulSoup(html, "html.parser")
        else:
            print(f"[Thordata] HTTP {res.status} for {url}")
    except Exception as e:
        print(f"[Thordata] Error: {e}")
    finally:
        conn.close()
    return None

# DIRECT FALLBACK
def scrape_direct(url: str) -> BeautifulSoup | None:
    import requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "nl-BE,nl;q=0.9",
        "Referer": "https://www.google.com/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            dbg = f"debug_direct_{url.split('//')[1].split('/')[0]}.html"
            with open(dbg, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"[Direct] Saved {dbg} ({len(r.text):,} chars)")
            return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"[Direct] Error: {e}")
    return None

# SCRAPE LISTINGS
def scrape_car_listings() -> list[dict]:
    urls = [
        "https://www.2dehands.be/l/auto-s/?page=1",
        "https://www.autoscout24.be/nl/lst/?page=1",
    ]
    all_cars = []

    for base_url in urls:
        print(f"\n--- Scraping {base_url} ---")
        soup = scrape_with_thordata(base_url) or scrape_direct(base_url)
        if not soup:
            print(f"WARNING: No HTML for {base_url}")
            continue

        listings = []

        if "2dehands" in base_url:
            listings = soup.select("article[data-testid='ad-card']") or \
                       soup.select("div.hz-Card") or \
                       soup.select("a[href*='/a/']")

        elif "autoscout" in base_url:
            listings = soup.select("article.cldt-summary-full-item") or \
                       soup.select("a[data-item-name='detail-page-link']")

        if not listings:
            print(f"WARNING: No listings on {base_url}")
            continue

        print(f"Found {len(listings)} listings")

        for item in listings[:50]:
            try:
                link = item if item.name == "a" else item.find("a", href=True)
                if not link: continue

                url = link.get("href", "")
                if url.startswith("/"):
                    url = ("https://www.autoscout24.be" + url
                           if "autoscout" in base_url
                           else "https://www.2dehands.be" + url)

                title_tag = link.select_one("h2, [data-testid='ad-title'], .ListItem_title__ndA4A")
                title = title_tag.get_text(strip=True) if title_tag else ""

                raw_text = item.get_text(" ", strip=True)

                # BRAND
                brand = "Unknown"
                if "autoscout" in base_url:
                    brand = item.get("data-make", "").strip()
                if not brand or brand == "Unknown":
                    brand_match = re.search(r"\b(VW|Volkswagen|BMW|Audi|Mercedes|Opel|Peugeot|Ford|Citroen|Renault|Toyota|Fiat|Skoda|Nissan|Volvo|Seat|Hyundai|Kia|Mazda|Honda|Suzuki|Porsche|Land Rover|Lexus|Tesla|Mini|Jeep|Alfa Romeo|Dacia)\b", title, re.I)
                    brand = brand_match.group(1) if brand_match else "Unknown"

                # MODEL
                model = title
                if brand != "Unknown":
                    model = re.sub(rf"\b{re.escape(brand)}\b", "", title, count=1, flags=re.I).strip()
                model = re.sub(r"\+.*|Meer voertuigen.*", "", model).strip()

                all_cars.append({
                    "title": title,
                    "brand": brand,
                    "model": model,
                    "url": url,
                    "raw_text": raw_text,
                })
            except Exception as e:
                print(f"Parse error: {e}")
                continue

        print(f"Collected {len(all_cars)} cars so far")
        time.sleep(2)

    return all_cars


# PARSE FINAL DATA
def parse_car_data(cars: list[dict]) -> list[dict]:
    print("\n=== PARSING FINAL DATA ===")
    premium = {"BMW", "Mercedes", "Audi", "Porsche", "Volvo", "Land Rover", "Lexus", "Tesla"}
    now = datetime.now().year
    parsed = []

    for i, car in enumerate(cars):
        txt = car["raw_text"]

        price_match = re.search(r"€\s*([\d.,]+)", txt)
        price_numeric = float(price_match.group(1).replace(".", "").replace(",", ".")) if price_match else 0.0

        year_match = re.search(r"(20[0-2]\d|19[89]\d)", txt)
        year_numeric = int(year_match.group(1)) if year_match else 0

        km_match = re.search(r"([\d.]+)\s*km", txt, re.I)
        mileage_numeric = int(km_match.group(1).replace(".", "")) if km_match else 0

        brand = car["brand"]
        model = car["model"] or "Unknown"
        age = now - year_numeric if year_numeric else 0
        is_premium = brand in premium

        match_confidence = sum([
            price_numeric > 0,
            year_numeric > 0,
            mileage_numeric > 0,
            brand != "Unknown"
        ])

        if price_numeric > 0:
            parsed.append({
                "model": model,
                "brand": brand,
                "price_numeric": price_numeric,
                "year_numeric": year_numeric,
                "mileage_numeric": mileage_numeric,
                "age": age,
                "is_premium": is_premium,
                "url": car["url"],
                "match_confidence": match_confidence,
                "title": car["title"]
            })
            print(f"[{i+1}] {brand} {model} | €{price_numeric:,.0f} | {year_numeric} | {mileage_numeric:,} km")

    print(f"\nTotal parsed: {len(parsed)} cars")
    return parsed

# SAVE JSON
def save_to_json(cars: list[dict]) -> None:
    with open("cars_data.json", "w", encoding="utf-8") as f:
        json.dump(cars, f, ensure_ascii=False, indent=4)
    print(f"\nSAVED cars_data.json ({len(cars)} cars)")


def job():
    print("\n" + "="*80)
    print(f"SCRAPING STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Sites: 2dehands.be & autoscout24.be")
    print("="*80)

    raw = scrape_car_listings()
    if raw:
        parsed = parse_car_data(raw)
        if parsed:
            save_to_json(parsed)
            print("SCRAPING & SAVING COMPLETED")
        else:
            print("No valid cars after parsing")
    else:
        print("No listings collected")

    print("="*80 + "\n")

# 8. RUN + SCHEDULE
if __name__ == "__main__":
    job()

    schedule.every(7).days.at("03:00").do(job)
    print("Scheduler started. Next run in 7 days at 03:00.")
    print("Keep this script running (or use Task Scheduler/cron).")

    while True:
        schedule.run_pending()
        time.sleep(60)
