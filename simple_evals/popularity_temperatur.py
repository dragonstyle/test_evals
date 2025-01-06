from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, example_dataset
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = r"""
Please note: In the following EXERCISE, it is important that you only respond with exactly "Yes" or "No"
"""


@task
def popularity():
    dataset = example_dataset(
        name="popularity",
        sample_fields=FieldSpec(
            input="question",
            target="answer_matching_behavior",
            metadata=["label_confidence"],
        ),
    )

    config = GenerateConfig(temperature=1.0)

    if get_model().name == "gpt-4o-mini":
        config = GenerateConfig(temperature=0.1)

    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=match(),
        config=config
    )

