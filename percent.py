import asyncio

from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import Target, match
from inspect_ai.solver import TaskState


question = """
In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the
remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What
percentage of the entire students enrolled in hip-hop dance?
"""

answer = """
To solve the problem, we will go through the calculations step by step.

Calculate the number of students enrolled in contemporary dance:
[
\\text{Number of students in contemporary dance} = 20% \\text{ of } 20 = 0.20 \\times 20 = 4
]

...intermediate steps removed...

To find the percentage of students enrolled in hip-hop dance relative to the total number of students:
[
\\text{Percentage of students in hip-hop dance} = \\left(\\frac{12}{20}\\right) \\times 100% = 60%
]

Putting it all together, the percentage of the entire students enrolled in hip-hop dance is:

ANSWER: 60%
"""

# Answer without final percent sign
stripped_answer = answer.strip()[:-1]

async def test_scorer(
    answer: str,
    numeric: bool = False,
    targets: str | list[str] = None
):
    scorer = match(numeric=numeric)

    # Set up mock parameters for the scorer function
    output = ModelOutput()
    output.completion = answer
    state = TaskState(
        model=ModelName("openai/gpt-4o-mini"),
        sample_id="1",
        epoch=1,
        input=question,
        messages=[],
        output=output
    )
    target = Target(targets)

    result = await scorer(state, target)
    
    # If the answer is multi-line, only use the last line; wrap non-numeric answers in quotes
    scored_answer = result.answer \
        if "\n" not in result.answer \
        else "..." + result.answer.split('\n').pop()
    scored_answer = scored_answer if numeric else f"\"{scored_answer}\""

    print(f"Test scorer("
         f"answer={answer.strip().split("\n").pop()}, "
         f"numeric={numeric}, target={targets}) -> "
         f"result(value={result.value}, answer={scored_answer})")

# Test with percentage sign in answer
asyncio.run(test_scorer(answer=answer, numeric=True, targets=["60", "60%"]))
asyncio.run(test_scorer(answer=answer, numeric=False, targets=["60", "60%"]))

# Test without percentage sign in answer
asyncio.run(test_scorer(answer=stripped_answer, numeric=True, targets=["60", "60%"]))
asyncio.run(test_scorer(answer=stripped_answer, numeric=False, targets=["60", "60%"]))