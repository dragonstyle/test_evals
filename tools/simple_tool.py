

from inspect_ai import Task, task
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.dataset import Sample
from inspect_ai.solver import use_tools, generate

@tool
def get_top_song() -> Tool:

    async def execute(call_sign: str) -> str:
        """Returns the most popular song for the requested station.

        Args:
            call_sign (str): The call sign for the station for which you want
              the most popular song.

        Returns:
            response (json): The most popular song and artist.
        """
        song = ""
        artist = ""
        if call_sign == 'WZPZ':
            song = "Elemental Hotel"
            artist = "8 Storey Hike"

        else:
            raise ToolError(f"Station {call_sign} not found.")

        return f"{song}: {artist}"
    

    return execute



@task
def test_tools():
    return Task(
        dataset = [Sample(input="What is the most popular song on WZPZ?")],
        solver = [use_tools(get_top_song()), generate()]
    )