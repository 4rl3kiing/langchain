"""Tests for ToolCacheMiddleware functionality."""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain.agents.factory import create_agent
from langchain.agents.middleware import ToolCacheMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel

CALLS: list[str] = []


@tool
def lookup(query: str) -> str:
    """Read-only lookup that records every real execution."""
    CALLS.append(query)
    return f"result for {query}"


@tool
def other_tool(query: str) -> str:
    """Tool that is not listed for caching."""
    CALLS.append(f"other:{query}")
    return f"other {query}"


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


def _request(name: str, args: dict[str, Any], call_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call=ToolCall(name=name, args=args, id=call_id),
        tool=None,
        state={"messages": []},
        runtime=None,  # type: ignore[arg-type]
    )


def _ok(call_id: str, content: str = "value") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name="lookup", id=f"id-{call_id}")


def test_requires_at_least_one_tool() -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        ToolCacheMiddleware(tools=[])


def test_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        ToolCacheMiddleware(tools=["lookup"], max_entries=0)
    with pytest.raises(ValueError, match="ttl_seconds"):
        ToolCacheMiddleware(tools=["lookup"], ttl_seconds=0)


def test_second_identical_call_is_served_from_cache() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[str] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(request.tool_call["id"] or "")
        return _ok(request.tool_call["id"] or "")

    first = cache.wrap_tool_call(_request("lookup", {"q": "a"}, "1"), handler)
    second = cache.wrap_tool_call(_request("lookup", {"q": "a"}, "2"), handler)

    assert runs == ["1"]
    assert isinstance(first, ToolMessage)
    assert isinstance(second, ToolMessage)
    assert second.content == first.content
    # The cached copy must answer the new call and must not reuse the stored message id.
    assert second.tool_call_id == "2"
    assert second.id is None
    assert first.tool_call_id == "1"


def test_argument_order_does_not_change_the_key() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    cache.wrap_tool_call(_request("lookup", {"a": 1, "b": 2}, "1"), handler)
    cache.wrap_tool_call(_request("lookup", {"b": 2, "a": 1}, "2"), handler)
    assert len(runs) == 1


def test_different_arguments_are_not_shared() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "1"), handler)
    cache.wrap_tool_call(_request("lookup", {"q": "b"}, "2"), handler)
    assert len(runs) == 2


def test_unlisted_tools_bypass_the_cache() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    cache.wrap_tool_call(_request("other_tool", {"q": "a"}, "1"), handler)
    cache.wrap_tool_call(_request("other_tool", {"q": "a"}, "2"), handler)
    assert len(runs) == 2


def test_accepts_tool_instances() -> None:
    cache = ToolCacheMiddleware(tools=[lookup])
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    cache.wrap_tool_call(_request("lookup", {"query": "a"}, "1"), handler)
    cache.wrap_tool_call(_request("lookup", {"query": "a"}, "2"), handler)
    assert len(runs) == 1


def test_error_results_are_not_cached() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return ToolMessage(
            content="boom", tool_call_id=request.tool_call["id"] or "", status="error"
        )

    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "1"), handler)
    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "2"), handler)
    assert len(runs) == 2


def test_command_results_are_not_cached() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[int] = []

    def handler(_request: ToolCallRequest) -> Command[Any]:
        runs.append(1)
        return Command(update={"x": 1})

    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "1"), handler)
    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "2"), handler)
    assert len(runs) == 2


def test_entries_expire_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("langchain.agents.middleware.tool_cache.time.monotonic", lambda: now[0])
    cache = ToolCacheMiddleware(tools=["lookup"], ttl_seconds=10)
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "1"), handler)
    now[0] += 9
    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "2"), handler)
    assert len(runs) == 1
    now[0] += 2
    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "3"), handler)
    assert len(runs) == 2


def test_least_recently_used_entry_is_evicted() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"], max_entries=2)
    runs: list[str] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(request.tool_call["args"]["q"])
        return _ok(request.tool_call["id"] or "")

    def call(q: str, call_id: str) -> None:
        cache.wrap_tool_call(_request("lookup", {"q": q}, call_id), handler)

    call("a", "1")
    call("b", "2")
    call("a", "3")  # hit, makes "a" the most recently used
    call("c", "4")  # evicts "b"
    call("a", "5")  # still cached
    call("b", "6")  # was evicted, runs again
    assert runs == ["a", "b", "c", "b"]


def test_custom_key_function() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"], key_func=lambda name, args: f"{name}:{args['q']}")
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    cache.wrap_tool_call(_request("lookup", {"q": "a", "noise": 1}, "1"), handler)
    cache.wrap_tool_call(_request("lookup", {"q": "a", "noise": 2}, "2"), handler)
    assert len(runs) == 1


def test_clear_empties_the_cache() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[int] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "1"), handler)
    cache.clear()
    cache.wrap_tool_call(_request("lookup", {"q": "a"}, "2"), handler)
    assert len(runs) == 2


async def test_async_path_uses_the_cache() -> None:
    cache = ToolCacheMiddleware(tools=["lookup"])
    runs: list[int] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        runs.append(1)
        return _ok(request.tool_call["id"] or "")

    first = await cache.awrap_tool_call(_request("lookup", {"q": "a"}, "1"), handler)
    second = await cache.awrap_tool_call(_request("lookup", {"q": "a"}, "2"), handler)
    assert len(runs) == 1
    assert isinstance(first, ToolMessage)
    assert isinstance(second, ToolMessage)
    assert second.tool_call_id == "2"


def test_agent_runs_a_repeated_tool_call_once() -> None:
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "lookup", "args": {"query": "x"}, "id": "1"}],
            [{"name": "lookup", "args": {"query": "x"}, "id": "2"}],
            [{"name": "other_tool", "args": {"query": "x"}, "id": "3"}],
            [{"name": "other_tool", "args": {"query": "x"}, "id": "4"}],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[lookup, other_tool],
        middleware=[ToolCacheMiddleware(tools=["lookup"])],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("go")]},
        {"configurable": {"thread_id": "t"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    # Every tool call still gets its own, correctly linked, ToolMessage...
    assert [m.tool_call_id for m in tool_messages] == ["1", "2", "3", "4"]
    # ...but the cached tool only ran once while the uncached one ran both times.
    assert CALLS == ["x", "other:x", "other:x"]
