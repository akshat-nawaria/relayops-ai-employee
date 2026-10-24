
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create the RelayOps backend application.
app = FastAPI(title="RelayOps API")

# Find the main project folder and our company information files.
BASE_DIR = Path(__file__).resolve().parent
LEADS_FILE = BASE_DIR / "data" / "leads.json"
SOP_FILE = BASE_DIR / "data" / "company_sop.md"

# Ollama runs our AI model locally on this computer.
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:4b"

# Connect directly to Ollama instead of routing local requests through a proxy.
OLLAMA_OPENER = build_opener(ProxyHandler({}))


# Define the format of a question sent to the AI endpoint.
class AskRequest(BaseModel):
    # Accept a question between 1 and 2,000 characters.
    question: str = Field(min_length=1, max_length=2000)


# Read customer records from the JSON file and handle file errors in one place.
def load_leads() -> list[dict[str, Any]]:
    try:
        # Open the file and convert its JSON contents into Python data.
        with LEADS_FILE.open("r", encoding="utf-8") as file:
            leads = json.load(file)

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

    # Make sure the file contains a list of customer records.
    if not isinstance(leads, list) or not all(
        isinstance(lead, dict) for lead in leads
    ):
        raise HTTPException(
            status_code=500,
            detail="The leads data file must contain a list of customer records."
        )

    return leads


# Check whether the RelayOps backend is running.
@app.get("/")
def home():
    return {
        "message": "RelayOps backend is running",
        "status": "ok"
    }


# Return all customer records.
@app.get("/leads")
def get_leads():
    # Load the records and return them with their total count.
    leads = load_leads()

    return {
        "count": len(leads),
        "leads": leads
    }


# Find one customer using their unique lead ID.
@app.get("/leads/{lead_id}")
def get_lead(lead_id: str):
    # Read the customer records before searching.
    leads = load_leads()

    # Return the record whose ID matches the requested ID.
    for lead in leads:
        if lead.get("id") == lead_id:
            return lead

    # Return a 404 error if the requested customer does not exist.
    raise HTTPException(
        status_code=404,
        detail=f"Lead '{lead_id}' was not found."
    )


# Read and return the company's Standard Operating Procedure.
@app.get("/company-sop")
def get_company_sop():
    try:
        # Read the actual rules from the Markdown document.
        content = SOP_FILE.read_text(encoding="utf-8")

        # Return both the filename and the document contents.
        return {
            "file": SOP_FILE.name,
            "content": content
        }

    except FileNotFoundError:
        # Report a clear error if the company rules document is missing.
        raise HTTPException(
            status_code=500,
            detail="The company SOP file was not found."
        )


# Send a question to the local AI model and return its answer.
@app.post("/ask")
def ask_ai(request_data: AskRequest):
    # Remove unnecessary spaces and reject a blank question.
    question = request_data.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a question."
        )

    # Prepare the instructions and question that Qwen3 will receive.
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the AI component of RelayOps, an early-stage "
                    "AI employee prototype. Answer clearly and honestly. "
                    "Do not invent facts or claim that you performed actions "
                    "you did not perform. This endpoint provides no tools "
                    "for changing records or taking external actions."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    # Convert the request into JSON so Ollama can understand it.
    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        # Send the request directly to Ollama, bypassing the network proxy.
        # The longer timeout allows the local model time to generate an answer.
        with OLLAMA_OPENER.open(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as error:
        # Report a useful error if Ollama rejects the request.
        details = error.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {error.code}: {details[:500]}"
        ) from error

    except (URLError, TimeoutError, OSError) as error:
        # Explain when the local Ollama service cannot be reached.
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to Ollama. Check that Ollama is running "
                "and that the local API is available."
            )
        ) from error

    except json.JSONDecodeError as error:
        # Handle a response that is not valid JSON.
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid JSON response."
        ) from error

    # Check that Ollama returned the expected response structure.
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unexpected response."
        )

    message = result.get("message")

    if not isinstance(message, dict):
        raise HTTPException(
            status_code=502,
            detail="Ollama's response did not contain a message."
        )

    # Extract the final answer generated by the model.
    answer = message.get("content")

    if not isinstance(answer, str) or not answer.strip():
        # Do not report success if the model did not produce an answer.
        raise HTTPException(
            status_code=502,
            detail="The AI model did not return a usable answer."
        )

    # Return the model name and its answer to the requester.
    return {
        "model": MODEL_NAME,
        "answer": answer.strip()
    }
