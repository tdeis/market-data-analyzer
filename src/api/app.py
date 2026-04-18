from fastapi import FastAPI
from orchestrator import run_research
from src.db.database import ExperimentDB

app = FastAPI()
db = ExperimentDB()


# =========================
# RUN EXPERIMENT
# =========================
@app.post("/run")
def run(tickers: list[str]):

    output = run_research(tickers)

    return {"leaderboard": output["leaderboard"].to_dict(), "db": db.leaderboard()}


# =========================
# GET HISTORY
# =========================
@app.get("/history")
def history():

    return db.leaderboard()


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():

    return {"status": "ok"}
