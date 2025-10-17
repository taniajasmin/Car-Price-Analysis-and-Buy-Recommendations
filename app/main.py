from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
# import threading
# from .ai_calculations import schedule_scraping

app = FastAPI(title="Car Profit API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.include_router(router)

# # Start scraping scheduler safely
# def start_scheduler():
#     try:
#         threading.Thread(target=schedule_scraping, daemon=True).start()
#         print("Scraping scheduler started")
#     except Exception as e:
#         print(f"Failed to start scheduler: {str(e)}")

# start_scheduler()





# threading.Thread(target=schedule_scraping, daemon=True).start()