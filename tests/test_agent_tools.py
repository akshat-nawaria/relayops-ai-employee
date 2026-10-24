
import unittest

# Import the functions we want to test.
from agent_tools import execute_tool, get_company_sop, get_lead


class TestAgentTools(unittest.TestCase):

    # A valid customer ID should return the correct record.
    def test_get_existing_customer(self):
        result = get_lead("L-105")

        self.assertEqual(result["id"], "L-105")
        self.assertEqual(result["name"], "Meera Sharma")
        self.assertIsNone(result["budget_inr"])

    # An unknown customer ID should return a clear error.
    def test_get_missing_customer(self):
        result = get_lead("L-999")

        self.assertIn("error", result)
        self.assertIn("L-999", result["error"])

    # The SOP tool should return the actual company rules.
    def test_get_company_sop(self):
        result = get_company_sop()

        self.assertEqual(result["file"], "company_sop.md")
        self.assertIn("Data Accuracy", result["content"])
        self.assertIn("Never invent", result["content"])

    # A tool that is not approved must be rejected.
    def test_unknown_tool_is_rejected(self):
        result = execute_tool("delete_all_records", {})

        self.assertIn("error", result)
        self.assertIn("not allowed", result["error"])

    # A customer lookup without an ID must be rejected.
    def test_missing_lead_id_is_rejected(self):
        result = execute_tool("get_lead", {})

        self.assertIn("error", result)
        self.assertIn("lead_id", result["error"])

    # Extra arguments must not be silently accepted.
    def test_unexpected_lead_arguments_are_rejected(self):
        result = execute_tool(
            "get_lead",
            {
                "lead_id": "L-105",
                "delete_record": True
            }
        )

        self.assertIn("error", result)
        self.assertIn("Unexpected arguments", result["error"])

    # Tool arguments must be supplied as a dictionary.
    def test_invalid_argument_type_is_rejected(self):
        result = execute_tool("get_lead", ["L-105"])

        self.assertIn("error", result)
        self.assertIn("JSON object", result["error"])

    # The SOP tool must not receive unexpected arguments.
    def test_sop_unexpected_arguments_are_rejected(self):
        result = execute_tool(
            "get_company_sop",
            {"lead_id": "L-105"}
        )

        self.assertIn("error", result)
        self.assertIn("accepts no arguments", result["error"])


if __name__ == "__main__":
    # Run the tests when this file is executed directly.
    unittest.main()
