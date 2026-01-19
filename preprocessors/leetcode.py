"""
LeetCode dataset preprocessor for coding tasks.
"""

from typing import Dict, Any, Optional
import signal


class TimeoutError(Exception):
    """Raised when test execution exceeds timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Test execution exceeded timeout")


def validate_code_solution(solution_code: str, test_code: str, entry_point: str,
                          prompt_helpers: str, timeout_seconds: int = 2) -> bool:
    """
    Validate that the code solution passes all tests within the timeout.

    Args:
        solution_code: The solution code to test
        test_code: The test code with check() function
        entry_point: Entry point (e.g., "Solution().methodName")
        prompt_helpers: Helper functions/imports from the prompt field
        timeout_seconds: Maximum time allowed for tests (default: 2 seconds)

    Returns:
        True if all tests pass within timeout, False otherwise
    """
    try:
        # Set up timeout alarm (Unix-only)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        namespace = {}

        # Execute helpers from prompt field
        exec(prompt_helpers, namespace)

        # Execute solution
        exec(solution_code, namespace)

        # Parse entry point
        if '().' in entry_point:
            class_name, method_name = entry_point.split('().')
            obj = namespace[class_name]()
            entry_point_func = getattr(obj, method_name)
        else:
            entry_point_func = namespace[entry_point]

        # Execute tests
        exec(test_code, namespace)
        namespace['check'](entry_point_func)

        # Cancel the alarm
        signal.alarm(0)

        return True

    except TimeoutError:
        # Test took too long
        signal.alarm(0)
        return False

    except Exception:
        # Test failed or had errors
        signal.alarm(0)
        return False


def preprocess_leetcode(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Preprocess LeetCode examples for GRPO training.
    Validates that the ground truth solution passes tests within 2 seconds.

    Args:
        example: Raw example from LeetCode dataset with fields:
            - problem_description: The coding problem statement
            - test: Test cases in check() function format
            - entry_point: Function to call (e.g., "Solution().twoSum")
            - completion: Ground truth solution
            - prompt: Helper functions and imports

    Returns:
        Processed example with question, answer, test_code, entry_point, helpers, and domain tag.
        Returns None if validation fails or times out (filters out ~6.6% of examples).
    """
    problem_description = example.get('problem_description', '')
    test_code = example.get('test', '')
    entry_point = example.get('entry_point', '')
    completion = example.get('completion', '')
    prompt_helpers = example.get('prompt', '')
    starter_code = example.get('starter_code', '')

    # Validate that the ground truth solution works and completes quickly (within 2 seconds)
    if not validate_code_solution(completion, test_code, entry_point, prompt_helpers, timeout_seconds=2):
        # Skip examples that fail validation or take too long
        # Based on testing: ~93.4% pass, ~6.6% filtered out
        return None

    return {
        'question': problem_description + "\n\n" + starter_code,
        'answer': completion,
        'test_code': test_code,
        'entry_point': entry_point,
        'helpers': prompt_helpers,  # Keep helpers for evaluation
        'domain': 'code',
    }
