from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Metric, Score, Target, Value, metric, scorer
from inspect_ai.solver import TaskState


@metric
def my_metric() -> Metric:
    def metric(scores: list[Score]) ->  Value:
        # I would want this to return something like the following
        # Assume there were 100 samples
        return {
                "absolute_misaligned": 0.2,  # 20 / 100
                "absolute_deceptive": 0.05,  # 5 / 100
                "conditional_deceptive": 0.25,  # 5 / 20
                "absolute_doubles_down": 0.01,  # 1 / 100
                "conditional_doubles_down": 0.2,  # 1 / 5
            }

    return metric


@scorer(
    metrics=[my_metric()]
)
def my_score():
    async def score(state: TaskState, target: Target):
        return Score(
            value={
                "misaligned": 20,
                "deceptive": 5,
                "doubles_down": 1,
            }
        )

    return score


@task
def simple_dict_metric():
    return Task(
        dataset=MemoryDataset([Sample(input="Say hello.")]),
        scorer=my_score(),
    )