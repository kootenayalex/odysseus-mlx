"""Tool-call lifting for MLX-served Qwen models (ported from Baton's gateway).

MLX backends emit tool calls as plain text in several shapes; the gateway
must lift them into OpenAI `tool_calls` and reject false positives (example
JSON the model writes as prose). These lock in every documented shape.
"""

from services.mlx_tool_calls import (
    classify_stream,
    extract_tool_calls,
    normalize_tool_calls,
    ollama_messages_to_openai,
    tool_names,
)

NAMES = ["read_file", "run_shell", "search"]


# --- the four documented emission shapes --------------------------------- #
def test_qwen3_tool_call_tag():
    content = 'Sure.<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "read_file"
    assert '"path": "a.py"' in tcs[0]["function"]["arguments"]
    assert cleaned == "Sure."


def test_tools_wrapped_json():
    content = '<tools>\n{"name": "run_shell", "arguments": {"cmd": "ls"}}\n</tools>'
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "run_shell"
    assert not cleaned  # only the call, nothing left


def test_bare_json_object():
    content = '{"name": "search", "arguments": {"q": "mlx"}}'
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "search"


def test_named_tag_form():
    content = "<read_file arguments='{\"path\": \"x.py\"}'/>"
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "read_file"
    assert '"path": "x.py"' in tcs[0]["function"]["arguments"]


def test_fenced_named_tag():
    content = "```xml\n<run_shell arguments='{\"cmd\": \"pwd\"}'/>\n```"
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "run_shell"


# --- false-positive rejection -------------------------------------------- #
def test_unoffered_name_rejected():
    # The model writes example JSON that looks like a call to a tool that
    # was never offered — must NOT be lifted.
    content = 'Here is an example: {"name": "delete_everything", "arguments": {}}'
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert tcs == []
    assert cleaned == content  # untouched


def test_no_names_allows_any_bare_call():
    # With no offered-names filter, a well-formed bare call is still lifted
    # (matches Baton: names filter is optional).
    content = '{"name": "anything", "arguments": {"a": 1}}'
    tcs, _ = extract_tool_calls(content)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "anything"


def test_plain_prose_untouched():
    content = "The file contains a function called read_file that opens a path."
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert tcs == []
    assert cleaned == content


def test_think_block_stripped_from_cleaned():
    content = '<think>I should read the file</think><tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
    tcs, cleaned = extract_tool_calls(content, NAMES)
    assert len(tcs) == 1
    assert not cleaned  # think + call both consumed


def test_braces_in_strings_dont_break_parsing():
    content = '{"name": "run_shell", "arguments": {"cmd": "echo \\"{not a call}\\""}}'
    tcs, _ = extract_tool_calls(content, NAMES)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "run_shell"


def test_multiple_calls():
    content = ('<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
               '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>')
    tcs, _ = extract_tool_calls(content, NAMES)
    assert [t["function"]["name"] for t in tcs] == ["read_file", "search"]


# --- normalize_tool_calls (non-streaming response shape) ----------------- #
def test_normalize_lifts_and_sets_finish_reason():
    resp = {"choices": [{"message": {"role": "assistant",
            "content": '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'},
            "finish_reason": "stop"}]}
    out = normalize_tool_calls(resp, NAMES)
    msg = out["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "search"
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_normalize_leaves_existing_tool_calls_alone():
    resp = {"choices": [{"message": {"role": "assistant", "content": "",
            "tool_calls": [{"id": "x", "type": "function", "function": {"name": "y", "arguments": "{}"}}]}}]}
    out = normalize_tool_calls(resp, NAMES)
    assert out["choices"][0]["message"]["tool_calls"][0]["id"] == "x"


# --- tool_names ----------------------------------------------------------- #
def test_tool_names_extraction():
    body = {"tools": [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "search"}},
        "garbage",
    ]}
    assert tool_names(body) == ["read_file", "search"]
    assert tool_names({}) == []


# --- classify_stream ------------------------------------------------------ #
def test_classify_prose():
    mode, safe = classify_stream("Hello there, this is prose.", NAMES)
    assert mode == "prose"
    assert safe == len("Hello there, this is prose.")


def test_classify_tool_start():
    mode, _ = classify_stream('<tool_call>{"name": "read', NAMES)
    assert mode == "tool"


def test_classify_bare_json_is_tool():
    mode, _ = classify_stream('{"name": "search"', NAMES)
    assert mode == "tool"


def test_classify_partial_tag_held():
    # A lone "<" might be the start of <tool_call> — hold, don't forward as prose.
    mode, _ = classify_stream("<", NAMES)
    assert mode == "scan"


def test_classify_think_streams_as_prose():
    mode, safe = classify_stream("<think>reasoning so far", NAMES)
    assert mode == "scan"
    assert safe == len("<think>reasoning so far")


def test_classify_named_tag_start():
    mode, _ = classify_stream("<read_file arguments=", NAMES)
    assert mode == "tool"


# --- ollama_messages_to_openai ------------------------------------------- #
def test_ollama_text_messages():
    out = ollama_messages_to_openai([{"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "hi"}]


def test_ollama_images_to_data_url():
    out = ollama_messages_to_openai([{"role": "user", "content": "look", "images": ["aGVsbG8="]}])
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
