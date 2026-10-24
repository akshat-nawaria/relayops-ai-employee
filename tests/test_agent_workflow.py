
import unittest
from unittest.mock import patch

from fastapi import HTTPException

# Import the request model and endpoint function we want to test.
from app import AskRequest, ask_ai


class TestAgentWorkflow(unittest.TestCase):

    # A clarification request must stop before any tools are executed.
    def test_clarification_stops_execution(self):
        plan = {
            "goal": "Send an email to a customer.",
            "steps": [],
            "needs_clarification": True,
            "clarification_question": (
                "Sending emails is not currently supported."
            )
        }

        with patch("app.create_plan", return_value=plan), \
             patch("app.call_ollama") as ai_call:
            result = ask_ai(
                AskRequest(question="Send an email to a customer.")
            )

        self.assertEqual(result["tools_used"], [])
        self.assertIn("not currently supported", result["answer"])
        ai_call.assert_not_called()

    # A planned customer lookup should execute and return an answer.
    def test_planned_tool_executes(self):
        plan = {
            "goal": "Look up customer L-105.",
            "steps": [
                {
                    "tool": "get_lead",
                    "arguments": {"lead_id": "L-105"},
                    "reason": "Retrieve the requested customer."
                }
            ],
            "needs_clarification": False,
            "clarification_question": ""
        }

        # Simulate Qwen3 requesting the approved customer lookup.
        tool_response = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_lead",
                            "arguments": {"lead_id": "L-105"}
                        }
                    }
                ]
            }
        }

        # Simulate Qwen3's final answer after receiving the tool result.
        final_response = {
            "message": {
                "content": "L-105 belongs to Meera Sharma at Greenfield Foods."
            }
        }

        with patch("app.create_plan", return_value=plan), \
             patch(
                 "app.call_ollama",
                 side_effect=[tool_response, final_response]
             ):
            result = ask_ai(
                AskRequest(question="Look up customer L-105.")
            )

        self.assertEqual(result["tools_used"], ["get_lead"])
        self.assertIn("Meera Sharma", result["answer"])

    # A tool that is missing from the approved plan must not execute.
    def test_unplanned_tool_is_rejected(self):
        plan = {
            "goal": "Look up customer L-105.",
            "steps": [
                {
                    "tool": "get_lead",
                    "arguments": {"lead_id": "L-105"},
                    "reason": "Retrieve the requested customer."
                }
            ],
            "needs_clarification": False,
            "clarification_question": ""
        }

        unplanned_response = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_company_sop",
                            "arguments": {}
                        }
                    }
                ]
            }
        }

        final_response = {
            "message": {
                "content": "The requested plan could not be completed."
            }
        }

        with patch("app.create_plan", return_value=plan), \
             patch(
                 "app.call_ollama",
                 side_effect=[unplanned_response, final_response]
             ), \
             patch("app.execute_tool") as tool_executor:

            with self.assertRaises(HTTPException) as error:
                ask_ai(
                    AskRequest(question="Look up customer L-105.")
                )

        self.assertEqual(error.exception.status_code, 502)
        tool_executor.assert_not_called()

    # If planning fails, RelayOps must stop before using any tools.
    def test_planner_failure_stops_execution(self):
        with patch(
            "app.create_plan",
            side_effect=RuntimeError("Ollama unavailable")
        ), patch("app.call_ollama") as ai_call, \
             patch("app.execute_tool") as tool_executor:

            with self.assertRaises(HTTPException) as error:
                ask_ai(
                    AskRequest(question="Look up customer L-105.")
                )

        self.assertEqual(error.exception.status_code, 503)
        ai_call.assert_not_called()
        tool_executor.assert_not_called()

    # The AI must not report success before required tools have run.
    def test_final_answer_before_tool_execution_is_rejected(self):
        plan = {
            "goal": "Look up customer L-105.",
            "steps": [
                {
                    "tool": "get_lead",
                    "arguments": {"lead_id": "L-105"},
                    "reason": "Retrieve the requested customer."
                }
            ],
            "needs_clarification": False,
            "clarification_question": ""
        }

        # Simulate the model answering without requesting the planned tool.
        premature_response = {
            "message": {
                "content": "The customer lookup is complete."
            }
        }

        with patch("app.create_plan", return_value=plan), \
             patch(
                 "app.call_ollama",
                 return_value=premature_response
             ), \
             patch("app.execute_tool") as tool_executor:

            with self.assertRaises(HTTPException) as error:
                ask_ai(
                    AskRequest(question="Look up customer L-105.")
                )

        self.assertEqual(error.exception.status_code, 502)
        tool_executor.assert_not_called()


if __name__ == "__main__":
    # Run these tests directly when needed.
    unittest.main()
