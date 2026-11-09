"""Tool result caching middleware for agents."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool


def _default_key(tool_name: str, args: dict[str, Any]) -> str:
    """Build a stable cache key from a tool name and its arguments."""
    return f"{tool_name}:{json.dumps(args, sort_keys=True, default=repr)}"


class ToolCacheMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Middleware that caches the results of idempotent tool calls.

    When the model calls the same tool with the same arguments more than once, for
    example while re-checking a fact across several turns, the cached result is
    returned instead of running the tool again. This saves latency and the cost of
    paid or rate-limited APIs.

    Caching is opt-in: you must name the tools to cache, because only side-effect-free
    tools are safe to memoize. Error results and `Command` results are never cached.

    The cache lives on the middleware instance and is shared by every run that uses
    it, including concurrent ones. It is in-memory and not persisted.

    Examples:
        !!! example "Cache a read-only lookup for five minutes"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ToolCacheMiddleware

            agent = create_agent(
                model,
                tools=[search_docs],
                middleware=[ToolCacheMiddleware(tools=["search_docs"], ttl_seconds=300)],
            )
            ```

        !!! example "Custom cache key"

            ```python
            # Ignore the `request_id` argument when deciding whether two calls match.
            cache = ToolCacheMiddleware(
                tools=["search_docs"],
                key_func=lambda name, args: f"{name}:{args.get('query')}",
            )
            ```
    """

    def __init__(
        self,
        *,
        tools: list[BaseTool | str],
        ttl_seconds: float | None = None,
        max_entries: int = 128,
        key_func: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        """Initialize `ToolCacheMiddleware`.

        Args:
            tools: Tools, or tool names, whose results may be cached.

                Only list tools that are idempotent and free of side effects.
            ttl_seconds: How long a cached result stays valid. `None` keeps entries
                until they are evicted.
            max_entries: Maximum number of cached results. The least recently used
                entry is evicted first.
            key_func: Optional function that maps a tool name and its arguments to a
                cache key. Defaults to a stable JSON encoding of both.

        Raises:
            ValueError: If `tools` is empty, `max_entries` is below 1, or `ttl_seconds`
                is not positive.
        """
        super().__init__()
        if not tools:
            msg = "`tools` must name at least one tool to cache."
            raise ValueError(msg)
        if max_entries < 1:
            msg = "`max_entries` must be at least 1."
            raise ValueError(msg)
        if ttl_seconds is not None and ttl_seconds <= 0:
            msg = "`ttl_seconds` must be positive when set."
            raise ValueError(msg)

        self._tool_names = {t if isinstance(t, str) else t.name for t in tools}
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._key_func = key_func or _default_key
        self._entries: OrderedDict[str, tuple[float, ToolMessage]] = OrderedDict()
        self._lock = threading.Lock()
        self.tools = []

    def clear(self) -> None:
        """Remove every cached result."""
        with self._lock:
            self._entries.clear()

    def _lookup(self, key: str, tool_call_id: str | None) -> ToolMessage | None:
        """Return a fresh copy of a cached result, or `None` on a miss or expiry."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, message = entry
            if self._ttl_seconds is not None and time.monotonic() - stored_at > self._ttl_seconds:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
        # The copy must answer the *current* tool call and must not reuse the stored
        # message `id`, otherwise `add_messages` would overwrite the earlier message.
        return message.model_copy(update={"tool_call_id": tool_call_id or "", "id": None})

    def _store(self, key: str, result: ToolMessage | Command[Any]) -> None:
        """Cache a successful `ToolMessage`; ignore errors and `Command` results."""
        if not isinstance(result, ToolMessage) or result.status == "error":
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), result)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def _key_for(self, request: ToolCallRequest) -> str | None:
        """Return the cache key for a request, or `None` if its tool is not cached."""
        name = request.tool_call["name"]
        if name not in self._tool_names:
            return None
        return self._key_func(name, request.tool_call["args"])

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Serve a cached result when available, otherwise run the tool and cache it.

        Args:
            request: Tool call request.
            handler: Callable that executes the tool.

        Returns:
            The cached or freshly produced tool result.
        """
        key = self._key_for(request)
        if key is None:
            return handler(request)
        cached = self._lookup(key, request.tool_call["id"])
        if cached is not None:
            return cached
        result = handler(request)
        self._store(key, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of `wrap_tool_call`.

        Args:
            request: Tool call request.
            handler: Async callable that executes the tool.

        Returns:
            The cached or freshly produced tool result.
        """
        key = self._key_for(request)
        if key is None:
            return await handler(request)
        cached = self._lookup(key, request.tool_call["id"])
        if cached is not None:
            return cached
        result = await handler(request)
        self._store(key, result)
        return result
