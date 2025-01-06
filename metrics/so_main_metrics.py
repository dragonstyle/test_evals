from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Metric, Score, Target, Value, metric, scorer
from inspect_ai.solver import TaskState


@metric
def nested_dict_metric() -> Metric:
    def metric(scores: list[Score]) ->  Value:

        total = 0
        for score in scores:
          if not isinstance(score.value, dict) and not isinstance(score.value, list):
            total = int(score.value)+ total

        return {
                "total": total,
                "ratio": total/26
            }

    return metric


@metric
def nested_array_metric() -> Metric:
    def metric(scores: list[Score]) ->  Value:

        total = 0
        for score in scores:
          if not isinstance(score.value, dict) and not isinstance(score.value, list):
            total = int(score.value)+ total

        return [total, total/26]

    return metric

@scorer(
    metrics={"*": [nested_dict_metric(), nested_array_metric()]}
)
def my_score():
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        return Score(
            value={
                "a": count_letter(answer, "a"),
                "b": count_letter(answer, "b"),
                "c": count_letter(answer, "c"),
                "d": count_letter(answer, "d"),
                "e": count_letter(answer, "e"),
                "f": count_letter(answer, "f"),
                "g": count_letter(answer, "g"),
                "h": count_letter(answer, "h"),
                "i": count_letter(answer, "i"),
                "j": count_letter(answer, "j"),
                "k": count_letter(answer, "k"),
                "l": count_letter(answer, "l"),
                "m": count_letter(answer, "m"),
                "n": count_letter(answer, "n"),
                "o": count_letter(answer, "o"),
                "p": count_letter(answer, "p"),
                "q": count_letter(answer, "q"),
                "r": count_letter(answer, "r"),
                "s": count_letter(answer, "s"),
                "t": count_letter(answer, "t"),
                "non_vowels": count_non_vowels(answer)
            }
        )

    return score


@task
def nested_dict_metrics() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input="Please say the coolest three words that you've ever heard.")]),
        scorer=my_score(),
    )

def count_letter(input_string: str, letter: str) -> int:
    return input_string.count(letter)

def count_non_vowels(input_string: str) -> int:
    vowels = "aeiouAEIOU"
    non_vowel_count = 0

    for char in input_string:
        if char.isalpha() and char not in vowels:
            non_vowel_count += 1

    return non_vowel_count