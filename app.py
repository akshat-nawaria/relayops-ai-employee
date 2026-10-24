
# FastAPI helps us build the backend that RelayOps will use.
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Create our RelayOps application.
app = FastAPI(title="RelayOps API")

# Find the main project folder, wherever it is stored on your computer.
BASE_DIR = Path(__file__).resolve().parent

# These are the files containing our customer data and company rules.
LEADS_FILE = BASE_DIR / "data" / "leads.json"
SOP_FILE = BASE_DIR / "data" / "company_sop.md"


# This endpoint checks whether our backend is working.
@app.get("/")
def home():
    return {
        "message": "RelayOps backend is running",
        "status": "ok"
    }


# This endpoint reads all the customer records from our JSON file.
@app.get("/leads")
def get_leads():
    try:
        # Open the file and convert its JSON data into Python data.
        with LEADS_FILE.open("r", encoding="utf-8") as file:
            leads = json.load(file)

        # Send the number of customers and their records back to the user.
        return {
            "count": len(leads),
            "leads": leads
        }

    except FileNotFoundError:
        # Tell the user if the customer data file is missing.
        raise HTTPException(
            status_code=500,
            detail="The leads data file was not found."
        )

    except json.JSONDecodeError:
        # Report a clear error if the JSON file has invalid formatting.
        raise HTTPException(
            status_code=500,
            detail="The leads data file contains invalid JSON."
        )


# This endpoint reads the company's rules from the SOP document.
@app.get("/company-sop")
def get_company_sop():
    try:
        # Read the entire SOP as ordinary text.
        content = SOP_FILE.read_text(encoding="utf-8")

        # Return the filename and the rules stored inside it.
        return {
            "file": SOP_FILE.name,
            "content": content
        }

    except FileNotFoundError:
        # Explain the problem if the SOP document cannot be found.
        raise HTTPException(
            status_code=500,
            detail="The company SOP file was not found."
        )
