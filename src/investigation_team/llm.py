"""
LLM calling, structured-output parsing, and repair.

Deliberately has NO dependency on google.colab — in the original
notebook, `os.environ['ANTHROPIC_API_KEY'] = userdata.get(...)` sat
directly in the same cell as this module's logic, which meant the
package could never be imported outside a Colab runtime. That
Colab-specific line now belongs in the notebook, before it imports
this package: the notebook pulls the key from userdata and sets a
plain environment variable; this module just expects that variable
to already be set, the same way it would be in any other deployment
(a `.env` file, a container's environment, a CI secret, etc.).
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional

import aisuite as ai
from langfuse import observe

# aisuite's Anthropic provider reads ANTHROPIC_API_KEY from the
# environment itself at call time. The caller (notebook or any other
# deployment) is responsible for setting it before importing this
# package — see the module docstring above.
Client = ai.Client()

DEFAULT_AGENT_MODEL = "anthropic:claude-haiku-4-5"

AGENT_MODEL_MAP = {
    "incident_framing_agent": DEFAULT_AGENT_MODEL,
    "hypothesis_generator_agent": DEFAULT_AGENT_MODEL,
    "evidence_planning_agent": DEFAULT_AGENT_MODEL,
    "evidence_test_creation_agent": DEFAULT_AGENT_MODEL,
    "evidence_analyst_agent": DEFAULT_AGENT_MODEL,
    "hypothesis_challenger_agent": DEFAULT_AGENT_MODEL,
    "root_cause_remediation_agent": DEFAULT_AGENT_MODEL,
}

# Fix vs. the original notebook: hypothesis_challenger_agent and
# root_cause_remediation_agent had no entries here and silently fell
# back to the 3000-token default in llm_call(). Given explicit
# entries for consistency and so future tuning is deliberate, not
# accidental.
AGENT_MAX_TOKENS = {
    "incident_framing_agent": 2500,
    "hypothesis_generator_agent": 2000,
    "evidence_planning_agent": 2500,       # output is now a short gap list
    "evidence_test_creation_agent": 5000,  # detailed but structurally capped at 8 items
    "evidence_analyst_agent": 2500,
    "hypothesis_challenger_agent": 2500,
    "root_cause_remediation_agent": 2500,
}

LLM_USAGE: List[Dict[str, Any]] = []


def trim_messages(messages, max_chars=12000):

    trimmed = []
    total_chars = 0

    # start from newest messages first
    for msg in reversed(messages):

        content = msg.get("content", "")

        if not isinstance(content, str):
            content = str(content)

        remaining = max_chars - total_chars

        if remaining <= 0:
            break

        # trim oversized message if needed
        if len(content) > remaining:
            content = content[-remaining:]

        trimmed.append({
            **msg,
            "content": content
        })

        total_chars += len(content)

    # restore chronological order
    return list(reversed(trimmed))


@observe(name="llm_call", as_type='generation')
def llm_call(
    agent_name: str,
    messages: list,
    temperature: float = 1.0,
    tools: list = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Observability wrapper that all agents use when calling the LLM.
    Logs model, input, output, and token usage to Langfuse.
    """

    selected_model = AGENT_MODEL_MAP.get(agent_name, DEFAULT_AGENT_MODEL)

    if max_tokens is None:
        max_tokens = AGENT_MAX_TOKENS.get(agent_name, 3000)

    messages = trim_messages(messages, max_chars=12000)

    call_kwargs = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        call_kwargs["tools"] = tools

    response = Client.chat.completions.create(**call_kwargs)
    content = response.choices[0].message.content

    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = input_tokens + output_tokens

    result = {
        "agent_name": agent_name,
        "model": selected_model,
        "temperature": temperature,
        "content": content,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        } if usage else {},
        "tools_available": bool(tools),
    }

    LLM_USAGE.append({
        "agent": agent_name,
        "model": selected_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    })

    print('llm call completed')

    return result


def clean_llm_dict_output(raw_output: str) -> str:
    cleaned = raw_output.strip()

    if not cleaned:
        raise ValueError(
            "Agent returned an empty response."
        )

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "Agent did not return the required dictionary.\n\n"
            f"Raw output:\n{raw_output}"
        )

    return cleaned[start:end + 1]


def parse_agent_response(
    raw_output: str,
) -> Dict[str, Any]:

    cleaned = clean_llm_dict_output(raw_output)

    # Try JSON first -- models asked for "a Python dictionary" often
    # produce JSON anyway (true/false/null, double-quoted strings).
    # Fall back to ast.literal_eval for genuinely Python-flavored
    # output (single-quoted strings, bare True/False/None) that
    # isn't valid JSON. This accepts either convention rather than
    # relying on a prompt instruction to pick one.
    try:
        parsed = json.loads(cleaned)
    except (ValueError, json.JSONDecodeError) as json_exc:
        try:
            parsed = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError) as ast_exc:
            raise ValueError(
                "Unable to parse agent response as a Python "
                "dictionary or JSON object.\n\n"
                f"JSON parser said: {json_exc}\n"
                f"Python parser said: {ast_exc}\n\n"
                f"Cleaned output:\n{cleaned}"
            ) from ast_exc

    if not isinstance(parsed, dict):
        raise TypeError(
            "Agent response must evaluate to a Python dictionary."
        )

    return parsed


def repair_agent_dict_response(
    raw_output: str,
    expected_schema: str,
    agent_name: str,
) -> Dict[str, Any]:
    """
    Repair an LLM response that failed the structured-output contract.

    The repair call performs formatting only. It must preserve the
    original analysis and convert it into the required dictionary.
    """

    repair_prompt = f"""
You are a structured-output formatter.

The previous agent returned an answer that violated its required
output format.

Your ONLY job is to convert the supplied response into a Python
dictionary matching the required schema.

Do not:
- add new reasoning;
- add new evidence;
- determine root cause;
- recommend remediation;
- include Markdown;
- include headings;
- include prose outside the dictionary.

If the original response used a different structure entirely (nested
phases, groups, or any wrapper other than the schema's own top-level
key), FLATTEN it: extract only the fields the schema actually asks
for, for each individual item, and discard every other field
(explanatory notes, phase groupings, discriminator descriptions,
rationale fields not in the schema, etc.) even if that means the
repaired response is substantially shorter than the original.


For hypothesis status, ONLY these values are valid:

- supported
- weakened
- rejected
- proposed

Map equivalent language as follows:

confirmed -> supported
strongly supported -> supported
unsupported -> weakened
falsified -> rejected
inconclusive -> proposed
unresolved -> proposed

Required schema:

{expected_schema}

Original response:

{raw_output}

Return ONLY the Python dictionary.
"""

    repair_response = llm_call(
        agent_name=f"{agent_name}_format_repair",
        messages=[
            {
                "role": "system",
                "content": (
                    "You convert malformed agent output into "
                    "strict structured Python dictionaries."
                ),
            },
            {
                "role": "user",
                "content": repair_prompt,
            },
        ],
        temperature=0.0,
        max_tokens=AGENT_MAX_TOKENS.get(agent_name, 3000),
    )

    return parse_agent_response(
        repair_response["content"]
    )


def parse_or_repair_agent_response(
    raw_output: str,
    expected_schema: str,
    agent_name: str,
) -> Dict[str, Any]:
    """
    Parse structured agent output.

    If the model violates the output contract, make one formatting-only
    repair attempt and parse the repaired result.

    Preserve hypothesis_id values EXACTLY as they appear in the original
    response or required input. Never abbreviate them to H1, H2, H3, etc.
    """

    try:
        return parse_agent_response(
            raw_output
        )

    except ValueError:
        print(
            f"⚠ {agent_name} returned malformed structured output. "
            "Attempting format repair."
        )

        return repair_agent_dict_response(
            raw_output=raw_output,
            expected_schema=expected_schema,
            agent_name=agent_name,
        )
