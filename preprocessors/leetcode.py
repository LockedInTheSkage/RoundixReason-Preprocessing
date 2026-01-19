import os
import signal
import threading
import warnings


class CodeExecutionTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise CodeExecutionTimeout("Test execution exceeded timeout")


def silent_excepthook(args):
    pass


def validate_code_solution(solution_code, test_code, entry_point, prompt_helpers, timeout_seconds=2):
    old_excepthook = threading.excepthook
    threading.excepthook = silent_excepthook

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    old_stdout_fd = os.dup(1)
    os.dup2(devnull, 2)
    os.dup2(devnull, 1)
    os.close(devnull)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            namespace = {}
            exec(prompt_helpers, namespace)
            exec(solution_code, namespace)

            if '().' in entry_point:
                class_name, method_name = entry_point.split('().')
                obj = namespace[class_name]()
                entry_point_func = getattr(obj, method_name)
            else:
                entry_point_func = namespace[entry_point]

            exec(test_code, namespace)
            namespace['check'](entry_point_func)

            signal.alarm(0)
            return True

    except CodeExecutionTimeout:
        signal.alarm(0)
        return False

    except Exception:
        signal.alarm(0)
        return False

    finally:
        os.dup2(old_stderr_fd, 2)
        os.dup2(old_stdout_fd, 1)
        os.close(old_stderr_fd)
        os.close(old_stdout_fd)
        threading.excepthook = old_excepthook


def preprocess_leetcode(example):
    problem_description = example.get('problem_description', '')
    test_code = example.get('test', '')
    entry_point = example.get('entry_point', '')
    completion = example.get('completion', '')
    prompt_helpers = example.get('prompt', '')
    starter_code = example.get('starter_code', '')

    if not validate_code_solution(completion, test_code, entry_point, prompt_helpers, timeout_seconds=2):
        return None

    return {
        'question': problem_description + "\n\n" + starter_code,
        'answer': completion,
        'test_code': test_code,
        'entry_point': entry_point,
        'helpers': prompt_helpers,
        'domain': 'code',
    }
