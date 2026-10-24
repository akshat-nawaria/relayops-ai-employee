
# These tools help RelayOps read files and provide API responses.
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Create our RelayOps backend application.
app = FastAPI(title="RelayOps API")

# Find the project folder and the files containing our company information.
BASE_DIR = Path(__file__).resolve().parent
LEADS_FILE = BASE_DIR / "data" / "leads.json"
SOP_FILE = BASE_DIR / "data" / "company_sop.md"


# This endpoint checks whether the backend is running.
@app.get("/")
def home():
    return {
        "message": "RelayOps backend is running",
        "status": "ok"
    }


# This endpoint returns all customer records.
@app.get("/leads")
def get_leads():
    try:
        # Open the JSON file and convert its contents into Python data.
        with LEADS_FILE.open("r", encoding="utf-8") as file:
            leads = json.load(file)

        # Return the total number of customers and their records.
        return {
            "count": len(leads),
            "leads": leads
        }

    except FileNotFoundError:
        # Report a clear error if the customer data file is missing.
        raise HTTPException(
            status_code=500,
            detail="The leads data file was not found."
        )

    except json.JSONDecodeError:
        # Report a clear error if the JSON file is incorrectly formatted.
        raise HTTPException(
            status_code=500,
            detail="The leads data file contains invalid JSON."
        )


# This endpoint finds one customer using their unique lead ID.
@app.get("/leads/{lead_id}")
def get_lead(lead_id: str):
    try:
        # Read the customer records from the JSON file.
        with LEADS_FILE.open("r", encoding="utf-8") as file:
            leads = json.load(file)

    except FileNotFoundError:
        # We cannot search customers if the data file is missing.
        raise HTTPException(
            status_code=500,
            detail="The leads data file was not found."
        )

    except json.JSONDecodeError:
        # Stop if the customer data cannot be read correctly.
        raise HTTPException(
            status_code=500,
            detail="The leads data file contains invalid JSON."
        )

    # Look through the records until we find the requested customer.
    for lead in leads:
        if lead.get("id") == lead_id:
            return lead

    # Return a 404 error if no customer has the requested ID.
    raise HTTPException(
        status_code=404,
        detail=f"Lead '{lead_id}' was not found."
    )


# This endpoint reads the company's rules from its SOP document.
@app.get("/company-sop")
def get_company_sop():
    try:
        # Read the rules as text from the company SOP file.
        content = SOP_FILE.read_text(encoding="utf-8")

        # Return the filename and the rules to the requester.
        return {
            "file": SOP_FILE.name,
            "content": content
        }

    except FileNotFoundError:
        # Report an understandable error if the SOP file is missing.
        raise HTTPException(
            status_code=500,
            detail="The company SOP file was not found."
        )
