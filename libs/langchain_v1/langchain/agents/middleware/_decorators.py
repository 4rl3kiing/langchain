"""Decorators that build middleware from plain functions.

Split out of `types.py` to keep that module focused on the core middleware types.
Everything public here is re-exported from `langchain.agents.middleware.types` (lazily,
to avoid a circular import) and from `langchain.agents.middleware`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any, cast, overload

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT
from typing_extensions import TypeVar

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    JumpTo,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ResponseT,
    StateT,
    StateT_contra,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

_SyncCallableWithStateAndRuntime = Callable[
    [StateT_contra, Runtime[ContextT]], dict[str, Any] | Command[Any] | None
]
_AsyncCallableWithStateAndRuntime = Callable[
    [StateT_contra, Runtime[ContextT]], Awaitable[dict[str, Any] | Command[Any] | None]
]
_CallableWithStateAndRuntime = (
    _SyncCallableWithStateAndRuntime[StateT_contra, ContextT]
    | _AsyncCallableWithStateAndRuntime[StateT_contra, ContextT]
)

_SyncCallableReturningSystemMessage = Callable[[ModelRequest[ContextT]], str | SystemMessage]
_AsyncCallableReturningSystemMessage = Callable[
    [ModelRequest[ContextT]], Awaitable[str | SystemMessage]
]
_CallableReturningSystemMessage = (
    _SyncCallableReturningSystemMessage[ContextT] | _AsyncCallableReturningSystemMessage[ContextT]
)

# Sync/async signatures for `wrap_model_call` interception; see `@wrap_model_call`.
_SyncCallableReturningModelResponse = Callable[
    [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]],
    ModelCallResult,
]
_AsyncCallableReturningModelResponse = Callable[
    [
        ModelRequest[ContextT],
        Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ],
    Awaitable[ModelCallResult],
]
_CallableReturningModelResponse = (
    _SyncCallableReturningModelResponse[ContextT, ResponseT]
    | _AsyncCallableReturningModelResponse[ContextT, ResponseT]
)

# Sync/async signatures for `wrap_tool_call` interception; see `@wrap_tool_call`.
_SyncCallableReturningToolResponse = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command[Any]]],
    ToolMessage | Command[Any],
]
_AsyncCallableReturningToolResponse = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
    Awaitable[ToolMessage | Command[Any]],
]
_CallableReturningToolResponse = (
    _SyncCallableReturningToolResponse | _AsyncCallableReturningToolResponse
)


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def hook_config(
    *,
    can_jump_to: list[JumpTo] | None = None,
) -> Callable[[CallableT], CallableT]:
    """Decorator to configure hook behavior in middleware methods.

    Use this decorator on `before_model` or `after_model` methods in middleware classes
    to configure their behavior. Currently supports specifying which destinations they
    can jump to, which establishes conditional edges in the agent graph.

    Args:
        can_jump_to: Optional list of valid jump destinations.

            Can be:

            - `'tools'`: Jump to the tools node
            - `'model'`: Jump back to the model node
            - `'end'`: Jump to the end of the graph

    Returns:
        Decorator function that marks the method with configuration metadata.

    Examples:
        !!! example "Using decorator on a class method"

            ```python
            class MyMiddleware(AgentMiddleware):
                @hook_config(can_jump_to=["end", "model"])
                def before_model(self, state: AgentState) -> dict[str, Any] | None:
                    if some_condition(state):
                        return {"jump_to": "end"}
                    return None
            ```

        Alternative: Use the `can_jump_to` parameter in `before_model`/`after_model`
        decorators:

        ```python
        @before_model(can_jump_to=["end"])
        def conditional_middleware(state: AgentState) -> dict[str, Any] | None:
            if should_exit(state):
                return {"jump_to": "end"}
            return None
        ```
    """

    def decorator(func: CallableT) -> CallableT:
        if can_jump_to is not None:
            func.__can_jump_to__ = can_jump_to  # type: ignore[attr-defined]
        return func

    return decorator


@overload
def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `before_model` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
                context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided directly) or a
            decorator function that can be applied to a function it is wrapping.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage"

            ```python
            @before_model
            def log_before_model(state: AgentState, runtime: Runtime) -> None:
                print(f"About to call model with {len(state['messages'])} messages")
            ```

        !!! example "With conditional jumping"

            ```python
            @before_model(can_jump_to=["end"])
            def conditional_before_model(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        !!! example "With custom state schema"

        Use a custom state schema when your middleware needs to read or write additional
        state fields that aren't part of the default agent state.

            ```python
            @before_model(state_schema=MyCustomState)
            def custom_before_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_value"}
            ```

        !!! example "Streaming custom events before model call"

            Use `runtime.stream_writer` to emit custom events before each model invocation.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @before_model
            async def notify_model_call(state: AgentState, runtime: Runtime) -> None:
                '''Notify user before model is called.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Thinking...",
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                # cannot narrow this sync-or-async callable union.
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeModelMiddleware")
            )

            # `type(...)` builds the correct middleware subclass at runtime, but
            # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "abefore_model": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            # `iscoroutinefunction` narrows `func` at runtime, but type checkers
            # cannot narrow this sync-or-async callable union.
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeModelMiddleware"))

        # `type(...)` builds the correct middleware subclass at runtime, but
        # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "before_model": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `after_model` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage for logging model responses"

            ```python
            @after_model
            def log_latest_message(state: AgentState, runtime: Runtime) -> None:
                print(state["messages"][-1].content)
            ```

        !!! example "With custom state schema"

        Use a custom state schema when your middleware needs to read or write additional
        state fields that aren't part of the default agent state.

            ```python
            @after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
            def custom_after_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_after_model"}
            ```

        !!! example "Streaming custom events after model call"

            Use `runtime.stream_writer` to emit custom events after model responds.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @after_model
            async def notify_model_response(state: AgentState, runtime: Runtime) -> None:
                '''Notify user after model has responded.'''
                last_message = state["messages"][-1]
                has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Using tools..." if has_tool_calls else "Response ready!",
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
        # Extract can_jump_to from decorator parameter or from function metadata
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                # cannot narrow this sync-or-async callable union.
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

            # `type(...)` builds the correct middleware subclass at runtime, but
            # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "aafter_model": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            # `iscoroutinefunction` narrows `func` at runtime, but type checkers
            # cannot narrow this sync-or-async callable union.
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

        # `type(...)` builds the correct middleware subclass at runtime, but
        # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "after_model": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `before_agent` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided directly) or a
            decorator function that can be applied to a function it is wrapping.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage"

            ```python
            @before_agent
            def log_before_agent(state: AgentState, runtime: Runtime) -> None:
                print(f"Starting agent with {len(state['messages'])} messages")
            ```

        !!! example "With conditional jumping"

            ```python
            @before_agent(can_jump_to=["end"])
            def conditional_before_agent(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        !!! example "With custom state schema"

        Use a custom state schema when your middleware needs to read or write additional
        state fields that aren't part of the default agent state.

            ```python
            @before_agent(state_schema=MyCustomState)
            def custom_before_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "initialized_value"}
            ```

        !!! example "Streaming custom events"

            Use `runtime.stream_writer` to emit custom events during agent execution.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import before_agent, AgentState
            from langchain.messages import HumanMessage
            from langgraph.runtime import Runtime


            @before_agent
            async def notify_start(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent is starting.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Initializing agent session...",
                    }
                )
                # Perform prerequisite tasks here
                runtime.stream_writer({"type": "status", "message": "Agent ready!"})


            agent = create_agent(
                model="openai:gpt-5.2",
                tools=[...],
                middleware=[notify_start],
            )

            # Consume with stream_mode="custom" to receive events
            async for mode, event in agent.astream(
                {"messages": [HumanMessage("Hello")]},
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    print(f"Status: {event}")
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                # cannot narrow this sync-or-async callable union.
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeAgentMiddleware")
            )

            # `type(...)` builds the correct middleware subclass at runtime, but
            # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "abefore_agent": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            # `iscoroutinefunction` narrows `func` at runtime, but type checkers
            # cannot narrow this sync-or-async callable union.
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeAgentMiddleware"))

        # `type(...)` builds the correct middleware subclass at runtime, but
        # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "before_agent": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `after_agent` hook.

    Async version is `aafter_agent`.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage for logging agent completion"

            ```python
            @after_agent
            def log_completion(state: AgentState, runtime: Runtime) -> None:
                print(f"Agent completed with {len(state['messages'])} messages")
            ```

        !!! example "With custom state schema"

        Use a custom state schema when your middleware needs to read or write additional
        state fields that aren't part of the default agent state.

            ```python
            @after_agent(state_schema=MyCustomState, name="MyAfterAgentMiddleware")
            def custom_after_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "finalized_value"}
            ```

        !!! example "Streaming custom events on completion"

            Use `runtime.stream_writer` to emit custom events when agent completes.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @after_agent
            async def notify_completion(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent has completed.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Agent execution complete!",
                        "total_messages": len(state["messages"]),
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
        # Extract can_jump_to from decorator parameter or from function metadata
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                # cannot narrow this sync-or-async callable union.
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

            # `type(...)` builds the correct middleware subclass at runtime, but
            # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "aafter_agent": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            # `iscoroutinefunction` narrows `func` at runtime, but type checkers
            # cannot narrow this sync-or-async callable union.
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

        # `type(...)` builds the correct middleware subclass at runtime, but
        # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "after_agent": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def dynamic_prompt(
    func: _CallableReturningSystemMessage[ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def dynamic_prompt(
    func: None = None,
) -> Callable[
    [_CallableReturningSystemMessage[ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def dynamic_prompt(
    func: _CallableReturningSystemMessage[ContextT] | None = None,
) -> (
    Callable[
        [_CallableReturningSystemMessage[ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically generate system prompts for the model.

    This is a convenience decorator that creates middleware using `wrap_model_call`
    specifically for dynamic prompt generation. The decorated function should return
    a string that will be set as the system prompt for the model request.

    Args:
        func: The function to be decorated.

            Must accept: `request: ModelRequest` - Model request (contains state and
            runtime)

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:
        - `str` – The system prompt string to use for the model request
        - `SystemMessage` – A complete system message to use for the model request

    Examples:
        Basic usage with dynamic content:

        ```python
        @dynamic_prompt
        def my_prompt(request: ModelRequest) -> str:
            user_name = request.runtime.context.get("user_name", "User")
            return f"You are a helpful assistant helping {user_name}."
        ```

        Using state to customize the prompt:

        ```python
        @dynamic_prompt
        def context_aware_prompt(request: ModelRequest) -> str:
            msg_count = len(request.state["messages"])
            if msg_count > 10:
                return "You are in a long conversation. Be concise."
            return "You are a helpful assistant."
        ```

        Using with agent:

        ```python
        agent = create_agent(model, middleware=[my_prompt])
        ```
    """

    def decorator(
        func: _CallableReturningSystemMessage[ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest[ContextT],
                handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
            ) -> ModelResponse[Any] | AIMessage:
                prompt = await cast("_AsyncCallableReturningSystemMessage[ContextT]", func)(request)
                if isinstance(prompt, SystemMessage):
                    request = request.override(system_message=prompt)
                else:
                    request = request.override(system_message=SystemMessage(content=prompt))
                return await handler(request)

            middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

            # `type(...)` builds the correct middleware subclass at runtime, but
            # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": AgentState,
                        "tools": [],
                        "awrap_model_call": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse[Any]],
        ) -> ModelResponse[Any] | AIMessage:
            prompt = cast("_SyncCallableReturningSystemMessage[ContextT]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return handler(request)

        async def async_wrapped_from_sync(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any] | AIMessage:
            # Delegate to sync function
            prompt = cast("_SyncCallableReturningSystemMessage[ContextT]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return await handler(request)

        middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

        # `type(...)` builds the correct middleware subclass at runtime, but
        # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": AgentState,
                    "tools": [],
                    "wrap_model_call": wrapped,
                    "awrap_model_call": async_wrapped_from_sync,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_model_call(
    func: _CallableReturningModelResponse[ContextT, ResponseT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def wrap_model_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningModelResponse[ContextT, ResponseT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def wrap_model_call(
    func: _CallableReturningModelResponse[ContextT, ResponseT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningModelResponse[ContextT, ResponseT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Create middleware with `wrap_model_call` hook from a function.

    Converts a function with handler callback into middleware that can intercept model
    calls, implement retry logic, handle errors, and rewrite responses.

    Args:
        func: Function accepting (request, handler) that calls handler(request)
            to execute the model and returns `ModelResponse` or `AIMessage`.

            Request contains state and runtime.
        state_schema: Custom state schema.

            Defaults to `AgentState`.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            Defaults to function name.

    Returns:
        `AgentMiddleware` instance if func provided, otherwise a decorator.

    Examples:
        !!! example "Basic retry logic"

            ```python
            @wrap_model_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        !!! example "Model fallback"

            ```python
            @wrap_model_call
            def fallback_model(request, handler):
                # Try primary model
                try:
                    return handler(request)
                except Exception:
                    pass

                # Try fallback model
                request = request.override(model=fallback_model_instance)
                return handler(request)
            ```

        !!! example "Rewrite response content (full `ModelResponse`)"

            ```python
            @wrap_model_call
            def uppercase_responses(request, handler):
                response = handler(request)
                ai_msg = response.result[0]
                return ModelResponse(
                    result=[AIMessage(content=ai_msg.content.upper())],
                    structured_response=response.structured_response,
                )
            ```

        !!! example "Simple `AIMessage` return (converted automatically)"

            ```python
            @wrap_model_call
            def simple_response(request, handler):
                # AIMessage is automatically converted to ModelResponse
                return AIMessage(content="Simple response")
            ```
    """

    def decorator(
        func: _CallableReturningModelResponse[ContextT, ResponseT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest[ContextT],
                handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
            ) -> ModelCallResult[ResponseT]:
                return await cast(
                    "_AsyncCallableReturningModelResponse[ContextT, ResponseT]", func
                )(request, handler)

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapModelCallMiddleware")
            )

            # `type(...)` builds the correct middleware subclass at runtime, but
            # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "awrap_model_call": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
        ) -> ModelCallResult[ResponseT]:
            return cast("_SyncCallableReturningModelResponse[ContextT, ResponseT]", func)(
                request, handler
            )

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapModelCallMiddleware"))

        # `type(...)` builds the correct middleware subclass at runtime, but
        # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "wrap_model_call": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_tool_call(
    func: _CallableReturningToolResponse,
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def wrap_tool_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningToolResponse],
    AgentMiddleware[StateT, ContextT],
]: ...


def wrap_tool_call(
    func: _CallableReturningToolResponse | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningToolResponse],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Create middleware with `wrap_tool_call` hook from a function.

    Async version is `awrap_tool_call`.

    Converts a function with handler callback into middleware that can intercept
    tool calls, implement retry logic, monitor execution, and modify responses.

    Args:
        func: Function accepting (request, handler) that calls
            handler(request) to execute the tool and returns final `ToolMessage` or
            `Command`.

            Can be sync or async.
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            Defaults to function name.

    Returns:
        `AgentMiddleware` instance if func provided, otherwise a decorator.

    Examples:
        !!! example "Retry logic"

            ```python
            @wrap_tool_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        !!! example "Async retry logic"

            ```python
            @wrap_tool_call
            async def async_retry(request, handler):
                for attempt in range(3):
                    try:
                        return await handler(request)
                    except Exception:
                        if attempt == 2:
                            raise
            ```

        !!! example "Modify request"

            ```python
            @wrap_tool_call
            def modify_args(request, handler):
                modified_call = {
                    **request.tool_call,
                    "args": {
                        **request.tool_call["args"],
                        "value": request.tool_call["args"]["value"] * 2,
                    },
                }
                request = request.override(tool_call=modified_call)
                return handler(request)
            ```

        !!! example "Short-circuit with cached result"

            ```python
            @wrap_tool_call
            def with_cache(request, handler):
                if cached := get_cache(request):
                    return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                result = handler(request)
                save_cache(request, result)
                return result
            ```

        !!! example "With custom state schema"

        Use a custom state schema when your middleware needs to read or write additional
        state fields that aren't part of the default agent state.

            ```python
            @wrap_tool_call(state_schema=MyCustomState)
            def custom_wrap_tool_call(request, handler):
                return handler(request)
            ```
    """

    def decorator(
        func: _CallableReturningToolResponse,
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
            ) -> ToolMessage | Command[Any]:
                return await cast("_AsyncCallableReturningToolResponse", func)(request, handler)

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapToolCallMiddleware")
            )

            # `type(...)` builds the correct middleware subclass at runtime, but
            # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "awrap_tool_call": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            return cast("_SyncCallableReturningToolResponse", func)(request, handler)

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapToolCallMiddleware"))

        # `type(...)` builds the correct middleware subclass at runtime, but
        # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "wrap_tool_call": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator
