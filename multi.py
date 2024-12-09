from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Metric, Score, Target, metric, scorer, mean, stderr
from inspect_ai.solver import TaskState


@scorer(
    metrics={
        "a_count": [mean(), stderr()],
        "e_count": [mean(), stderr()]
    }
)
def letter_count():
    async def score(state: TaskState, target: Target):
        answer = state.output.completion
        a_count = answer.count("a")
        e_count = answer.count("e")
        return Score(
            value={"a_count": a_count, "e_count": e_count},
            answer=answer
        )

    return score

@task
def mytask() -> Task:
    return Task(
        dataset=[Sample(input="Tell me a story.")],
        scorer=letter_count()
    )