from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import basic_agent, prompt_template
from inspect_ai.tool import python
from inspect_ai.model import GenerateConfig


template_prompt = """
Please write a python program that uses print statements to draw a an ascii-art picture of a {prompt}. 
Once your program is complete, use the python tool to execute your program. Verify that your 
program runs correctly and produces a very nice representation of a {prompt}. 

Once you have a drawing of {prompt} that you are happy with, submit a copy of the {prompt} as your answer using the submit tool.
"""

@task
def test_task_tools():
  return Task(
        dataset=[Sample(input="car", target="car"), Sample(input="bike", target="bike"), Sample(input="circle", target="circle")],
        solver=basic_agent(
            init=prompt_template(template=template_prompt),
            tools=[python(timeout=180)],
            max_attempts=3,
        ),
        scorer=model_graded_qa(model='openai/gpt-4o-mini', instructions='Please check if the answer contains an ascii drawing of a {criterion}. Carefully judge if the work produce a high quality ascii art representation of a {crtieron}. If it does, please respond with GRADE: C in your message, otherwise with GRADE: I.'),
        sandbox="docker",
        max_messages=25,
        epochs=3
    )