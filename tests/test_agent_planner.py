
import copy
import unittest

# Import the function that checks AI-generated plans.
from agent_planner import validate_plan


class TestAgentPlanner(unittest.TestCase):

    # Start with a valid plan that can be reused in our tests.
    def make_valid_plan(self):
        return {
            "goal": "Look up customer L-105.",
            "steps": [
                {
                    "tool": "get_lead",
                    "arguments": {"lead_id": "L-105"},
                    "reason": "Retrieve the requested customer record."
                }
            ],
            "needs_clarification": False,
            "clarification_question": ""
        }

    # A correctly structured plan should be accepted.
    def test_valid_plan_is_accepted(self):
        plan = self.make_valid_plan()

        result = validate_plan(plan)

        self.assertEqual(result, plan)

    # An unknown tool must never be accepted into a plan.
    def test_unknown_tool_is_rejected(self):
        plan = self.make_valid_plan()
        plan["steps"][0]["tool"] = "delete_all_records"

        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_plan(plan)

    # A customer lookup must include the lead ID.
    def test_missing_lead_id_is_rejected(self):
        plan = self.make_valid_plan()
        plan["steps"][0]["arguments"] = {}

        with self.assertRaisesRegex(ValueError, "lead_id"):
            validate_plan(plan)

    # Unexpected arguments must not be accepted.
    def test_unexpected_arguments_are_rejected(self):
        plan = self.make_valid_plan()
        plan["steps"][0]["arguments"]["delete_record"] = True

        with self.assertRaisesRegex(ValueError, "lead_id"):
            validate_plan(plan)

    # A plan requiring clarification must not contain tool steps.
    def test_clarification_plan_cannot_contain_steps(self):
        plan = self.make_valid_plan()
        plan["needs_clarification"] = True
        plan["clarification_question"] = "Which customer do you mean?"

        with self.assertRaisesRegex(ValueError, "must not contain tool steps"):
            validate_plan(plan)

    # A plan needing clarification must explain what is missing.
    def test_empty_clarification_question_is_rejected(self):
        plan = self.make_valid_plan()
        plan["steps"] = []
        plan["needs_clarification"] = True
        plan["clarification_question"] = ""

        with self.assertRaisesRegex(ValueError, "state what needs clarification"):
            validate_plan(plan)

    # The planner must not generate an excessively long plan.
    def test_too_many_steps_are_rejected(self):
        plan = self.make_valid_plan()
        plan["steps"] = [
            copy.deepcopy(plan["steps"][0]) for _ in range(5)
        ]

        with self.assertRaisesRegex(ValueError, "between zero and four steps"):
            validate_plan(plan)

    # Every step must explain why its tool is needed.
    def test_missing_step_reason_is_rejected(self):
        plan = self.make_valid_plan()
        plan["steps"][0]["reason"] = ""

        with self.assertRaisesRegex(ValueError, "explain its purpose"):
            validate_plan(plan)


if __name__ == "__main__":
    # Run the planner tests when this file is executed directly.
    unittest.main()
