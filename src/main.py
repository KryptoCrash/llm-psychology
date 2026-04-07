try:
    from .utils import base_prompt, conformity_prompt, load_questions
except ImportError:
    from utils import base_prompt, conformity_prompt, load_questions


if __name__ == "__main__":
    questions = load_questions()
    print(base_prompt(questions[0]))
    print(conformity_prompt(questions[0], 3))