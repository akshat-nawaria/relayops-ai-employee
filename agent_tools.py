
import json
from pathlib import Path
from typing import Any

# Find the files containing our company's information.
BASE_DIR = Path(__file__).resolve().parent
LEADS_FILE = BASE_DIR / "data" / "leads.json"
SOP_FILE = BASE_DIR / "data" / "company_sop.md"


# Tell the AI which tools it is allowed to request.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_lead",
            "description": (
                "Retrieve one customer record using its exact lead ID. "
                "Use this when the user asks about a specific customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_id": {
                        "type": "string",
                        "description": "The customer ID, such as L-104 or L-105."
                    }
                },
                "required": ["lead_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_sop",
            "description": (
                "Read the company's Standard Operating Procedure. "
                "Use this when the user asks about company rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# Read the customer records from the JSON file.
def load_leads() -> list[dict[str, Any]]:
    with LEADS_FILE.open("r", encoding="utf-8") as file:
        leads = json.load(file)

    # Check that the file contains a list of customer records.
    if not isinstance(leads, list) or not all(
        isinstance(lead, dict) for lead in leads
    ):
        raise ValueError("The leads file must contain a list of records.")

    return leads


# Find one customer using the ID provided to the tool.
def get_lead(lead_id: str) -> dict[str, Any]:
    try:
        # Load the latest customer records before searching.
        leads = load_leads()

    except FileNotFoundError:
        return {"error": "The leads data file was not found."}

    except (json.JSONDecodeError, ValueError):
        return {"error": "The leads data file is invalid."}

    # Return the matching customer if the ID exists.
    for lead in leads:
        if lead.get("id") == lead_id:
            return lead

    # Tell the caller when no customer has that ID.
    return {"error": f"Lead '{lead_id}' was not found."}


# Read the company's rules from its SOP document.
def get_company_sop() -> dict[str, Any]:
    try:
        # Read the real document instead of guessing its contents.
        content = SOP_FILE.read_text(encoding="utf-8")

    except FileNotFoundError:
        return {"error": "The company SOP file was not found."}

    # Return the filename and the rules it contains.
    return {
        "file": SOP_FILE.name,
        "content": content
    }


# Keep a list of the Python functions the AI may request.
TOOL_REGISTRY = {
    "get_lead": get_lead,
    "get_company_sop": get_company_sop
}


# Check each tool request before running the requested function.
def execute_tool(name: str, arguments: Any) -> dict[str, Any]:
    # Reject any tool that is not on our approved list.
    if name not in TOOL_REGISTRY:
        return {"error": f"Tool '{name}' is not allowed."}

    # Tool arguments must be supplied as a JSON object.
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be a JSON object."}

    # Check the arguments for the customer lookup tool.
    if name == "get_lead":
        lead_id = arguments.get("lead_id")

        if not isinstance(lead_id, str) or not lead_id.strip():
            return {"error": "A valid lead_id is required."}

        # Reject unexpected arguments rather than silently ignoring them.
        if set(arguments) != {"lead_id"}:
            return {"error": "Unexpected arguments for get_lead."}

        return get_lead(lead_id.strip())

    # The SOP tool does not need any arguments.
    if name == "get_company_sop":
        if arguments:
            return {"error": "get_company_sop accepts no arguments."}

        return get_company_sop()

    # Keep a final safeguard if the tool registry changes in the future.
    return {"error": "The requested tool could not be executed."}
