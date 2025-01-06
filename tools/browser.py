from typing import Any
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate
from inspect_ai.solver._use_tools import use_tools
from inspect_ai.tool import web_browser


@task
def browser(foo: list[str] | None = None, bar: dict[str, Any] | None = None, keys: str | None = None):
    return Task(
        dataset=[
            Sample(
                input="Use the web browser tool to navigate to a very cool website on the internet. Then, report the 5 best links that appear on that page. Please be sure to use the tools that you are provided."
            )
        ],
        solver=[
            use_tools(web_browser()),
            generate(),
        ],
        scorer=includes(),
        sandbox="docker",
        epochs=1
    )


if __name__ == "__main__":
    # vary the system prompt
    tasks = [
        browser(foo=["Hello","World","This","Is","A","Long","Array","Value"], bar={"key": { "val": "val"}},keys="5fbab3f7-8c80-4ca6-bbb5-c629f6dc55f1, 5202bed1-d2bd-4dc2-95bc-487a68c0c601, db5d3bd4-ad29-47a9-b677-8f24a5549247, 7b4347a3-7fce-4dd2-9bb9-1ff74e6116d5, 81498743-a138-4a44-9f9a-97af913747e6, 59f52897-ca3b-455d-b8b5-c8e106905fe5, 5ece5c32-08b0-469b-bc1b-31a2cfce7597, 68434a79-8ea0-491a-8acb-4494a7d0df2b, c58328b5-64c5-449b-a3b3-cefe82092d51, bdd2d30d-0a45-4f70-b307-7f7b621dfe96")
    ]
    eval(tasks, model = "openai/gpt-4o-mini")