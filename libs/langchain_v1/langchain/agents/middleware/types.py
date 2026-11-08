"""Types for middleware and agents."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    cast,
)

# Needed as top level import for Pydantic schema generation on AgentState
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolCallWrapper
from langgraph.types import Command, TracePolicy, omit_payload
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime
    from langgraph.stream._mux import TransformerFactory

    from langchain.agents.middleware._decorators import (
        after_agent,
        after_model,
        before_agent,
        before_model,
        dynamic_prompt,
        hook_config,
        wrap_tool_call,
    )
    from langchain.agents.structured_output import ResponseFormat

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ExtendedModelResponse",
    "InputAgentState",
    "ModelCallResult",
    "ModelRequest",
    "ModelResponse",
    "OmitFromSchema",
    "OutputAgentState",
    "ResponseT",
    "StateT_co",
    "ToolCallRequest",
    "ToolCallWrapper",
    "TracePolicy",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "omit_payload",
    "wrap_tool_call",
]

JumpTo = Literal["tools", "model", "end"]
"""Destination to jump to when a middleware node returns."""

ResponseT = TypeVar("ResponseT", default=Any)


class _ModelRequestOverrides(TypedDict, total=False):
    """Possible overrides for `ModelRequest.override()` method."""

    model: BaseChatModel
    system_message: SystemMessage | None
    messages: list[AnyMessage]
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    model_settings: dict[str, Any]
    state: AgentState[Any]


@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    """Model request information for the agent.

    Type Parameters:
        ContextT: The type of the runtime context.

            Defaults to `None` if not specified.
    """

    model: BaseChatModel
    messages: list[AnyMessage]  # excluding system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        model: BaseChatModel,
        messages: list[AnyMessage],
        system_message: SystemMessage | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        tools: list[BaseTool | dict[str, Any]] | None = None,
        response_format: ResponseFormat[Any] | None = None,
        state: AgentState[Any] | None = None,
        runtime: Runtime[ContextT] | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        """Initialize `ModelRequest` with backward compatibility for `system_prompt`.

        Args:
            model: The chat model to use.
            messages: List of messages (excluding system prompt).
            tool_choice: Tool choice configuration.
            tools: List of available tools.
            response_format: Response format specification.
            state: Agent state.
            runtime: Runtime context.
            model_settings: Additional model settings.
            system_message: System message instance (preferred).
            system_prompt: System prompt string (deprecated, converted to `SystemMessage`).

        Raises:
            ValueError: If both `system_prompt` and `system_message` are provided.
        """
        # Handle system_prompt/system_message conversion and validation
        if system_prompt is not None and system_message is not None:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if system_prompt is not None:
            system_message = SystemMessage(content=system_prompt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.model = model
            self.messages = messages
            self.system_message = system_message
            self.tool_choice = tool_choice
            self.tools = tools if tools is not None else []
            self.response_format = response_format
            self.state = state if state is not None else {"messages": []}
            self.runtime = runtime  # type: ignore[assignment]
            self.model_settings = model_settings if model_settings is not None else {}

    @property
    def system_prompt(self) -> str | None:
        """Get system prompt text from system_message.

        Returns:
            The content of the system message if present, otherwise `None`.
        """
        if self.system_message is None:
            return None
        return self.system_message.text

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute with a deprecation warning.

        Direct attribute assignment on `ModelRequest` is deprecated. Use the
        `override()` method instead to create a new request with modified attributes.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        # Special handling for system_prompt - convert to system_message
        if name == "system_prompt":
            warnings.warn(
                "Direct attribute assignment to ModelRequest.system_prompt is deprecated. "
                "Use request.override(system_message=SystemMessage(...)) instead to create "
                "a new request with the modified system message.",
                DeprecationWarning,
                stacklevel=2,
            )
            if value is None:
                object.__setattr__(self, "system_message", None)
            else:
                object.__setattr__(self, "system_message", SystemMessage(content=value))
            return

        warnings.warn(
            f"Direct attribute assignment to ModelRequest.{name} is deprecated. "
            f"Use request.override({name}=...) instead to create a new request "
            f"with the modified attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, name, value)

    def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]:
        """Replace the request with a new request with the given overrides.

        Returns a new `ModelRequest` instance with the specified attributes replaced.

        This follows an immutable pattern, leaving the original request unchanged.

        Args:
            **overrides: Keyword arguments for attributes to override.

                Supported keys:

                - `model`: `BaseChatModel` instance
                - `system_prompt`: deprecated, use `system_message` instead
                - `system_message`: `SystemMessage` instance
                - `messages`: `list` of messages
                - `tool_choice`: Tool choice configuration
                - `tools`: `list` of available tools
                - `response_format`: Response format specification
                - `model_settings`: Additional model settings
                - `state`: Agent state dictionary

        Returns:
            New `ModelRequest` instance with specified overrides applied.

        Examples:
            !!! example "Create a new request with different model"

                ```python
                new_request = request.override(model=different_model)
                ```

            !!! example "Override system message (preferred)"

                ```python
                from langchain_core.messages import SystemMessage

                new_request = request.override(
                    system_message=SystemMessage(content="New instructions")
                )
                ```

            !!! example "Override multiple attributes"

                ```python
                new_request = request.override(
                    model=ChatOpenAI(model="gpt-5.5"),
                    system_message=SystemMessage(content="New instructions"),
                )
                ```

        Raises:
            ValueError: If both `system_prompt` and `system_message` are provided.
        """
        # Handle system_prompt/system_message conversion
        if "system_prompt" in overrides and "system_message" in overrides:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if "system_prompt" in overrides:
            system_prompt = cast("str | None", overrides.pop("system_prompt"))  # type: ignore[typeddict-item]
            if system_prompt is None:
                overrides["system_message"] = None
            else:
                overrides["system_message"] = SystemMessage(content=system_prompt)

        return replace(self, **overrides)


@dataclass
class ModelResponse(Generic[ResponseT]):
    """Response from model execution including messages and optional structured output.

    The result will usually contain a single `AIMessage`, but may include an additional
    `ToolMessage` if the model used a tool for structured output.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any` if not specified.
    """

    result: list[BaseMessage]
    """List of messages from model execution."""

    structured_response: ResponseT | None = None
    """Parsed structured output if `response_format` was specified, `None` otherwise."""


@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    """Model response with an optional 'Command' from 'wrap_model_call' middleware.

    Use this to return a 'Command' alongside the model response from a
    'wrap_model_call' handler. The command is applied as an additional state
    update after the model node completes, using the graph's reducers (e.g.
    'add_messages' for the 'messages' key).

    Because each 'Command' is applied through the reducer, messages in the
    command are **added alongside** the model response messages rather than
    replacing them. For non-reducer state fields, later commands overwrite
    earlier ones (outermost middleware wins over inner).

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to 'Any' if not specified.
    """

    model_response: ModelResponse[ResponseT]
    """The underlying model response."""

    command: Command[Any] | None = None
    """Optional command to apply as an additional state update."""


ModelCallResult = ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]
"""Return type for model call handlers.

Middleware can return either:

- `ModelResponse`: Full response with messages and optional structured output
- `AIMessage`: Simplified return for simple use cases
- `ExtendedModelResponse`: Response with an optional `Command` for additional state updates
    `goto`, `resume`, and `graph` are not yet supported on these commands.
    A `NotImplementedError` will be raised if you try to use them.
"""


@dataclass
class OmitFromSchema:
    """Annotation used to mark state attributes as omitted from input or output schemas."""

    input: bool = True
    """Whether to omit the attribute from the input schema."""

    output: bool = True
    """Whether to omit the attribute from the output schema."""


OmitFromInput = OmitFromSchema(input=True, output=False)
"""Annotation used to mark state attributes as omitted from input schema."""

OmitFromOutput = OmitFromSchema(input=False, output=True)
"""Annotation used to mark state attributes as omitted from output schema."""

PrivateStateAttr = OmitFromSchema(input=True, output=True)
"""Annotation used to mark state attributes as purely internal for a given middleware."""


class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]


class InputAgentState(TypedDict):
    """Input state schema for the agent."""

    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]


class OutputAgentState(TypedDict, Generic[ResponseT]):
    """Output state schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]


# Deprecated aliases kept for backwards compatibility with external consumers that
# imported the previously private names. Remove in a future release.
_InputAgentState = InputAgentState
_OutputAgentState = OutputAgentState


StateT = TypeVar("StateT", bound=AgentState[Any], default=AgentState[Any])
StateT_co = TypeVar("StateT_co", bound=AgentState[Any], default=AgentState[Any], covariant=True)
StateT_contra = TypeVar("StateT_contra", bound=AgentState[Any], contravariant=True)


class _DefaultAgentState(AgentState[Any]):
    """AgentMiddleware default state."""


class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.

    Type Parameters:
        StateT: The type of the agent state. Defaults to `AgentState[Any]`.
        ContextT: The type of the runtime context. Defaults to `None`.
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

    state_schema: type[StateT] = cast("type[StateT]", _DefaultAgentState)
    """The schema for state passed to the middleware nodes."""

    tools: Sequence[BaseTool]
    """Additional tools registered by the middleware."""

    trace_policy: TracePolicy | None = None
    """Optional trace policy for this middleware's hook spans (`wrap_model_call`/
    `wrap_tool_call` and the `before_*`/`after_*` node hooks).

    By default (`None`), hook spans are traced normally. Set a `TracePolicy` to shape
    what they record -- e.g. `TracePolicy(process_inputs=omit_payload)` to drop the
    conversation `messages`/`state` payload while keeping the span and its timing.
    Messages are still captured on the inner model-call span.
    """

    transformers: Sequence[TransformerFactory] = ()
    """Stream transformer factories registered by the middleware.

    Each entry is a scope-aware factory invoked as `factory(scope)` so every
    invocation receives a fresh instance. Factories are merged with the
    `transformers` argument of [`create_agent`][langchain.agents.create_agent]
    at graph compile time, after the `ToolCallTransformer` and before any
    user-supplied entries.
    """

    @property
    def name(self) -> str:
        """The name of the middleware instance.

        Defaults to the class name, but can be overridden for custom naming.
        """
        return self.__class__.__name__

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before agent execution.
        """

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the agent execution starts.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before agent execution.
        """

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before model call.
        """

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the model is called.

        Args:
            state: The agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before model call.
        """

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after model call.
        """

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the model is called.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after model call.
        """

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Intercept and control model execution via handler callback.

        Async version is `awrap_model_call`

        The handler callback executes the model request and returns a `ModelResponse`.
        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Callback that executes the model request and returns
                `ModelResponse`.

                Call this to execute the model.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            The model call result.

        Examples:
            !!! example "Retry on error"

                ```python
                def wrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```

            !!! example "Rewrite response"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    ai_msg = response.result[0]
                    return ModelResponse(
                        result=[AIMessage(content=f"[{ai_msg.content}]")],
                        structured_response=response.structured_response,
                    )
                ```

            !!! example "Error to fallback"

                ```python
                def wrap_model_call(self, request, handler):
                    try:
                        return handler(request)
                    except Exception:
                        return ModelResponse(result=[AIMessage(content="Service unavailable")])
                ```

            !!! example "Cache/short-circuit"

                ```python
                def wrap_model_call(self, request, handler):
                    if cached := get_cache(request):
                        return cached  # Short-circuit with cached result
                    response = handler(request)
                    save_cache(request, response)
                    return response
                ```

            !!! example "Simple `AIMessage` return (converted automatically)"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    # Can return AIMessage directly for simple cases
                    return AIMessage(content="Simplified response")
                ```
        """
        msg = (
            "Synchronous implementation of wrap_model_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_model_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Intercept and control async model execution via handler callback.

        The handler callback executes the model request and returns a `ModelResponse`.

        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

                Call this to execute the model.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            The model call result.

        Examples:
            !!! example "Retry on error"

                ```python
                async def awrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return await handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```
        """
        msg = (
            "Asynchronous implementation of awrap_model_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_model_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after agent execution.
        """

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after agent execution.
        """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool execution for retries, monitoring, or modification.

        Async version is `awrap_tool_call`

        Multiple middleware compose automatically (first defined = outermost).

        Exceptions propagate unless `handle_tool_errors` is configured on `ToolNode`.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.

                Access state via `request.state` and runtime via `request.runtime`.
            handler: `Callable` to execute the tool (can be called multiple times).

        Returns:
            `ToolMessage` or `Command` (the final result).

        The handler `Callable` can be invoked multiple times for retry logic.

        Each call to handler is independent and stateless.

        Examples:
            !!! example "Modify request before execution"

                ```python
                def wrap_tool_call(self, request, handler):
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

            !!! example "Retry on error (call handler multiple times)"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

            !!! example "Conditional retry based on response"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        result = handler(request)
                        if isinstance(result, ToolMessage) and result.status != "error":
                            return result
                        if attempt < 2:
                            continue
                        return result
                ```
        """
        msg = (
            "Synchronous implementation of wrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_tool_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept and control async tool execution via handler callback.

        The handler callback executes the tool call and returns a `ToolMessage` or
        `Command`. Middleware can call the handler multiple times for retry logic, skip
        calling it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.

                Access state via `request.state` and runtime via `request.runtime`.
            handler: Async callable to execute the tool and returns `ToolMessage` or
                `Command`.

                Call this to execute the tool.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            `ToolMessage` or `Command` (the final result).

        The handler `Callable` can be invoked multiple times for retry logic.

        Each call to handler is independent and stateless.

        Examples:
            !!! example "Async retry on error"

                ```python
                async def awrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = await handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

                ```python
                async def awrap_tool_call(self, request, handler):
                    if cached := await get_cache_async(request):
                        return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                    result = await handler(request)
                    await save_cache_async(request, result)
                    return result
                ```
        """
        msg = (
            "Asynchronous implementation of awrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_tool_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)


_DECORATORS = frozenset(
    {
        "after_agent",
        "after_model",
        "before_agent",
        "before_model",
        "dynamic_prompt",
        "hook_config",
        "wrap_model_call",
        "wrap_tool_call",
    }
)
"""Decorators that live in `_decorators` but stay importable from this module."""


def __getattr__(name: str) -> Any:
    """Lazily resolve decorators moved to `_decorators` (avoids a circular import)."""
    if name in _DECORATORS:
        from langchain.agents.middleware import _decorators  # noqa: PLC0415

        return getattr(_decorators, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
