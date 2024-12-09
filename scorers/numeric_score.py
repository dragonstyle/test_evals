from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState, basic_agent, prompt_template
from inspect_ai.scorer import scorer, accuracy, stderr, Target, Score, match




template_prompt = """
Please answer the following question with a single word that best answers the question:

{prompt}
"""

@task
def numeric_score():
  return Task(
        dataset=[
            Sample(input="What the most common type of ICE powered transportation in the United States?", target="car"), 
            Sample(input="What has two wheels, pedals, and people ride it?", target=["bike", "bicycle"]), 
            Sample(input="What is a small freshwater body of water called?", target="pond"),
            Sample(input="What word is spelled incorrectly in every single dictionary?", target="incorrectly"),
            Sample(input=" Everyone in the world needs it, but they usually give it without taking it. What is it?", target="advice"),
            Sample(input="This question has an intentionally wrong target. What is the coolest word?", target="jambalaya")
        ],
        solver=basic_agent(
            init=prompt_template(template=template_prompt)
        ),
        scorer=numeric_scorer(),
    )

@scorer(metrics=[accuracy(), stderr()])
def numeric_scorer():

    async def score(state: TaskState, target: Target):

        # check for correct
        answer = state.output.completion
        value = len(target.text)

        # return score
        return Score(
            value = value,
            answer=answer
        )

    return score

