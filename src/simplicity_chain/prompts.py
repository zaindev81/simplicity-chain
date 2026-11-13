"""Prompt templates for the code agent."""

from __future__ import annotations

from string import Template

from .tools import Tool

SYSTEM_PROMPT_TEMPLATE = """
You are an expert assistant who solves tasks by writing Python code.

You have access to the following tools, callable as ordinary Python functions from your code:
$tools

Authorized imports:
$authorized_imports

How to act:
1. Reply with exactly one Python code block per step, delimited by $open_tag and $close_tag.
2. Your code is executed in a persistent Python session. Variables defined in one step are
   available in later steps, so you can build on previous results.
3. Use print(...) to inspect intermediate values. Whatever your code prints, and the value of
   its last expression, are returned to you as the observation.
4. When the task is complete, call final_answer(result) exactly once. This ends the run.
   Do not put the final answer in print(...) only.
5. Do not import modules that are not listed above. Do not access files, the network, or the
   operating system unless a tool provides that capability.
6. Never invent observations: only the actual execution output is returned to you.
7. If your code raises an error, you will see the error message in the next observation.
   Read it, fix the code, and try again.
8. Keep actions small and focused. You may use loops and helper functions inside the block.

Example:
$open_tag
numbers = [1, 2, 3]
print(numbers)
final_answer(sum(numbers))
$close_tag

$custom_instructions
""".strip()


def format_tools(tools: dict[str, Tool] | list[Tool]) -> str:
    """Render tools as a readable list of signatures for the system prompt."""
    values = list(tools.values()) if isinstance(tools, dict) else list(tools)
    if not values:
        return "No tools are available. Use plain Python only."
    lines = []
    for tool in values:
        parameters = []
        for parameter_name, schema in tool.inputs.items():
            type_name = schema.get("type", "any")
            if schema.get("nullable"):
                type_name = f"{type_name} | None"
            parameters.append(f"{parameter_name}: {type_name}")
        signature = f"{tool.name}({', '.join(parameters)}) -> {tool.output_type}"
        lines.append(f"- {signature}: {tool.description}")
    return "\n".join(lines)


def build_system_prompt(
    tools: dict[str, Tool] | list[Tool],
    authorized_imports: list[str],
    instructions: str | None = None,
) -> str:
    """Fill the system prompt template for a code agent."""
    if "*" in authorized_imports:
        imports_text = "Any installed package may be imported."
    else:
        imports_text = ", ".join(authorized_imports) or "None (no imports allowed)."

    custom_instructions = ""
    if instructions:
        custom_instructions = f"Additional instructions from the user:\n{instructions.strip()}"

    return Template(SYSTEM_PROMPT_TEMPLATE).substitute(
        tools=format_tools(tools),
        authorized_imports=imports_text,
        open_tag="```python",
        close_tag="```",
        custom_instructions=custom_instructions,
    )
