
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="RelayOps API")

BASE_DIR = Path(__file__).resolve().parent
LEADS_FILE = BASE_DIR / "data" / "leads.json"


@app.get("/")
def home():
    return {
        "message": "RelayOps backend is running",
        "status": "ok"
    }


@app.get("/leads")
def get_leads():
    try:
        with LEADS_FILE.open("r", encoding="utf-8") as file:
            leads = json.load(file)

        return {
            "count": len(leads),
            "leads": leads
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="The leads data file was not found."
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="The leads data file contains invalid JSON."
        )
