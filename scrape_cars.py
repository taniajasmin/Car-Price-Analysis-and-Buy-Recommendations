import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re
import json
import os
import schedule
import argparse
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("THORDATA_API_KEY")
API_URL = "https://universalapi.thordata.com/request"

if API_KEY and isinstance(API_KEY, str) and len(API_KEY) > 5:
    print("ThorData Key Loaded:", API_KEY[:5] + "...")
else:
    print("ThorData Key NOT FOUND or invalid → using direct requests fallback.")


# Utilities for extraction
CURRENT_YEAR = datetime.now().year

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def clean_number_str(s):
    """Remove currency symbols, dots, spaces and commas, return digits only string"""
    if not s:
        return ""
    return re.sub(r"[^\d]", "", s)

def extract_price(text):
    """
    Prefer explicit '€' pattern, fallback to first reasonable number > 100 (to avoid matching years).
    Returns float (euros) or None.
    """
    if not text:
        return None

    # Try euro sign pattern first
    m = re.search(r"€\s*([0-9\.\,\s]+)", text)
    if m:
        num_str = clean_number_str(m.group(1))
        if num_str:
            try:
                return float(num_str)
            except:
                pass

    # Fallback: find numbers and pick the largest (likely price) but ignore very small numbers < 50
    numbers = re.findall(r"(\d[\d\.\, ]{1,})", text.replace("\xa0", " "))
    candidates = []
    for n in numbers:
        nclean = clean_number_str(n)
        if nclean:
            try:
                val = float(nclean)
                if val >= 50:  # ignore tiny numbers (likely years/short info)
                    candidates.append(val)
            except:
                pass
    if candidates:
        # choose the maximum reasonable number
        return float(max(candidates))
    return None

def extract_year(text):
    """
    Extract 4-digit year in reasonable range (1950..CURRENT_YEAR+1)
    """
    if not text:
        return None
    years = re.findall(r"\b(19[5-9]\d|20[0-4]\d|20[5-9]\d)\b", text)  # years like 1950-2099 (filter later)
    for y in years:
        y_int = safe_int(y)
        if y_int and 1950 <= y_int <= CURRENT_YEAR + 1:
            return y_int
    # Another attempt: patterns like 'bouwjaar: 2018' or 'Eerste inschrijving: 05/10/23' (extract two-digit year)
    m = re.search(r"bouwjaar[:\s]*([0-9]{4})", text, re.IGNORECASE)
    if m:
        return safe_int(m.group(1))
    return None

def extract_mileage(text):
    """
    Look for km / kilometers / kms patterns.
    Return integer (km) or None.
    """
    if not text:
        return None
    # Common patterns: '123.456 km', '12 345 km', '123456 km', 'Km-stand: 237 000km'
    m = re.search(r"([0-9]{1,3}(?:[.\s][0-9]{3})+|[0-9]{3,6})\s*(?:km|kms|kilometer|kilometers)\b", text, re.IGNORECASE)
    if m:
        n = clean_number_str(m.group(1))
        return safe_int(n)
    # fallback: "km" adjacent digits
    m2 = re.search(r"(\d{1,6})\s*km\b", text, re.IGNORECASE)
    if m2:
        return safe_int(m2.group(1))
    return None

def extract_brand(title):
    """
    Simple heuristic: brand is usually the first token of the title.
    We normalize to capitalized string.
    """
    if not title:
        return None
    # remove leading punctuation
    t = title.strip()
    t = re.sub(r"^[^\w]+", "", t)
    first = t.split()[0]
    # remove non-alpha chars
    first_clean = re.sub(r"[^A-Za-z0-9\-]", "", first)
    if first_clean:
        return first_clean.capitalize()
    return None

def compute_is_premium(title, brand, price_numeric):
    """
    Heuristic for premium:
     - Brand in premium list OR
     - Title contains premium tokens OR
     - Price above threshold (e.g. > 50k)
    Returns 1 or 0
    """
    premium_brands = {"Mercedes", "Bmw", "Audi", "Porsche", "Lamborghini", "Ferrari", "Mclaren", "Rolls-Royce", "Bentley"}
    premium_tokens = ["AMG", "M SPORT", "M-Sport", "M SPORT", "S Line", "RS", "GT", "GTR", "Competition", "GTS", "Racing"]
    if brand and brand.capitalize() in premium_brands:
        return 1
    if title:
        t_up = title.upper()
        for tk in premium_tokens:
            if tk.upper() in t_up:
                return 1
    try:
        if price_numeric and float(price_numeric) >= 50000:
            return 1
    except:
        pass
    return 0

# Fetch HTML (ThorData optional, fallback to requests)
def fetch_html(url):
    # If ThorData key provided, use it so JS-rendered pages can be fetched
    if API_KEY and len(API_KEY) > 5:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        payload = {"url": url, "type": "html", "js_render": "True"}
        try:
            r = requests.post(API_URL, data=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json().get("html", "")
            else:
                print("ThorData error:", r.status_code, r.text[:200])
                return ""
        except Exception as e:
            print("ThorData request failed:", e)
            return ""
    # Fallback: direct GET with a User-Agent
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            return r.text
        else:
            print("Request error:", r.status_code)
            return ""
    except Exception as e:
        print("Requests GET failed:", e)
        return ""

# Extractors for sites
def extract_2dehands(html):
    soup = BeautifulSoup(html, "html.parser")
    listings = []

    for block in soup.find_all("a", href=True):
        href = block["href"]
        # heuristics to only pick actual listing anchors
        if not ("/v/" in href or "/l/" in href):
            continue

        if not href.startswith("http"):
            href = "https://www.2dehands.be" + href

        title_tag = block.find(["h2", "h3", "strong"])
        title = title_tag.get_text(strip=True) if title_tag else block.get_text(strip=True)[:150]

        image_tag = block.find("img")
        image = image_tag.get("src") if image_tag else None

        text = block.get_text(strip=True, separator=" ")

        # Core fields (raw)
        price = extract_price(text)
        year = extract_year(text + " " + title)
        mileage = extract_mileage(text)
        brand = extract_brand(title)

        # compute derived fields
        age = None
        if year:
            age = max(0, CURRENT_YEAR - int(year))
        is_premium = compute_is_premium(title, brand, price)

        # ensure we only save decent entries (title + url)
        if not title or len(title) < 2:
            continue

        listings.append({
            "title": title,
            "url": href,
            "raw_text": text[:800],
            "image_url": image,
            "price_numeric": price,
            "year_numeric": year,
            "mileage_numeric": mileage,
            "brand": brand,
            "age": age,
            "is_premium": is_premium,
            "source": "2dehands.be"
        })

    # dedupe by url
    unique = {item["url"]: item for item in listings}
    return list(unique.values())

def extract_autoscout(html):
    soup = BeautifulSoup(html, "html.parser")
    listings = []

    # CSS selector may change; try known selector then fallback
    cards = soup.select("a.css-xb5nz8")  # anchor used in some layouts
    if not cards:
        # fallback: anchors that have /nl/aanbod/ or /nl/lst maybe the listing links
        cards = soup.find_all("a", href=re.compile(r"/nl/aanbod/|/nl/lst|/nl/auto/|/en/used/"))

    for card in cards:
        href = card.get("href")
        if not href:
            continue
        if not href.startswith("http"):
            href = "https://www.autoscout24.be" + href

        title_tag = card.find(["h2", "h3"])
        title = title_tag.get_text(strip=True) if title_tag else card.get_text(strip=True)[:150]

        image_tag = card.find("img")
        image = image_tag.get("src") if image_tag else None

        text = card.get_text(strip=True, separator=" ")

        price = extract_price(text)
        year = extract_year(text + " " + title)
        mileage = extract_mileage(text)
        brand = extract_brand(title)
        age = None
        if year:
            age = max(0, CURRENT_YEAR - int(year))
        is_premium = compute_is_premium(title, brand, price)

        # Only include if we have at least title+url
        if not title or len(title) < 2:
            continue

        listings.append({
            "title": title,
            "url": href,
            "raw_text": text[:800],
            "image_url": image,
            "price_numeric": price,
            "year_numeric": year,
            "mileage_numeric": mileage,
            "brand": brand,
            "age": age,
            "is_premium": is_premium,
            "source": "autoscout24.be"
        })

    unique = {item["url"]: item for item in listings}
    return list(unique.values())


# Generic scraper
def scrape(base_url, pages, extractor, hard_limit=None):
    all_items = []

    if hard_limit is not None:
        pages = min(pages, hard_limit)

    for p in range(1, pages + 1):
        url = base_url.format(page=p)
        print(f"  Page {p}: {url}")

        html = fetch_html(url)
        if not html:
            print("   -> EMPTY HTML")
            continue

        results = extractor(html)
        print(f"   -> extracted {len(results)} cars")

        for r in results:
            r["page"] = p
            r["scraped_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        all_items.extend(results)
        time.sleep(1.0)  # polite pause

    return all_items

# Main run
def run_scraper(pages_2dh=2, pages_auto=5):
    print("        SCRAPE START")
    print("Started:", datetime.now())

    two = scrape(
        base_url="https://www.2dehands.be/l/auto-s/p/{page}/",
        pages=pages_2dh,
        extractor=extract_2dehands
    )
    print("2dehands extracted", len(two), "car listings")

    auto = scrape(
        base_url="https://www.autoscout24.be/nl/lst?page={page}",
        pages=pages_auto,
        extractor=extract_autoscout,
        hard_limit=10  # hard-limit requested
    )
    print("AutoScout extracted", len(auto), "car listings")

    merged = two + auto

    if merged:
        with open("cars_data.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print("\nSaved", len(merged), "cars to cars_data.json")
    else:
        print("\nNo cars scraped -- keeping old data")

    print("============================\n")
    print("         SCRAPE END")
    

# Scheduler
def start_scheduler():
    print("Scheduler active → scraping every 7 days.")
    schedule.every(7).days.do(run_scraper)
    # initial run
    print("\nInitial scrape starting now...")
    run_scraper()
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Scraper", add_help=True)
    parser.add_argument("--once", action="store_true", help="Run scraper once now")
    parser.add_argument("--pages-2dh", type=int, default=2, help="Pages to scrape from 2dehands")
    parser.add_argument("--pages-auto", type=int, default=5, help="Pages to scrape from AutoScout")
    args, unknown = parser.parse_known_args()

    if args.once:
        run_scraper(pages_2dh=args.pages_2dh, pages_auto=args.pages_auto)
    else:
        start_scheduler()
