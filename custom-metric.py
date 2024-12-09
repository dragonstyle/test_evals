from collections import Counter
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Metric, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState

@metric
def is_string() -> Metric:
    """Demonstrates that a string arrives on the scene."""

    def metric(scores: list[Score]) -> float:
        string_count = 0
        for score in scores:
            if isinstance(score.value, str):
                string_count = string_count + 1
        return string_count / len(scores)
    return metric



@scorer(metrics=[is_string()])
def letter_count() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        letter, frequency = most_common_letter(answer)
        return Score(
            value=letter,
            explanation=f"Letter {letter} appears {frequency} times" ,
            answer=answer
        )

    return score

@task
def custom_metric() -> Task:
    return Task(
        dataset=[Sample(input="Write a haiku about the moon."), Sample(input="Write a haiku about Saturn.")],
        scorer=letter_count()
    )


def most_common_letter(s):
    # Remove non-alphabetic characters and convert to lowercase
    filtered_string = [char.lower() for char in s if char.isalpha()]
    
    if not filtered_string:
        return None  # Return None if no letters are found

    # Use Counter to count the occurrences of each letter
    counter = Counter(filtered_string)
    
    # Find the most common letter
    most_common = counter.most_common(1)[0]
    
    return most_common[0], most_common[1]
