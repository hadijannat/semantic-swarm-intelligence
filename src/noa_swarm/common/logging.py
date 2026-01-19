"""Structured JSON logging with correlation IDs using loguru.

Provides centralized logging configuration with:
- JSON format for production
- Correlation ID injection for request tracing
- Request context tracking
- Configurable log levels
"""

from __future__ import annotations

import contextvars
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from loguru import Record

# Context variable for correlation ID - thread-safe and async-safe
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for additional request context
request_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "request_context", default={}
)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context.

    Returns:
        The correlation ID if set, None otherwise.
    """
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set or generate a correlation ID in the current context.

    Args:
        correlation_id: Optional correlation ID to set. If None, generates a new UUID.

    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    correlation_id_var.set(None)


def get_request_context() -> dict[str, Any]:
    """Get the current request context.

    Returns:
        Dictionary of request context values.
    """
    return request_context_var.get()


def set_request_context(context: dict[str, Any]) -> None:
    """Set the request context.

    Args:
        context: Dictionary of context values to set.
    """
    request_context_var.set(context)


def update_request_context(**kwargs: Any) -> None:
    """Update the request context with additional values.

    Args:
        **kwargs: Key-value pairs to add to the context.
    """
    current = request_context_var.get()
    request_context_var.set({**current, **kwargs})


def clear_request_context() -> None:
    """Clear the request context."""
    request_context_var.set({})


def json_serializer(record: Record) -> str:
    """Serialize log record to JSON format.

    Args:
        record: The loguru record to serialize.

    Returns:
        JSON string representation of the log record.
    """
    correlation_id = get_correlation_id()
    request_context = get_request_context()

    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["name"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add correlation ID if present
    if correlation_id:
        log_entry["correlation_id"] = correlation_id

    # Add request context if present
    if request_context:
        log_entry["context"] = request_context

    # Add exception info if present
    if record["exception"]:
        exception = record["exception"]
        log_entry["exception"] = {
            "type": exception.type.__name__ if exception.type else None,
            "value": str(exception.value) if exception.value else None,
            "traceback": exception.traceback is not None,
        }

    # Add extra fields from record
    extra = record.get("extra", {})
    if extra:
        # Filter out internal loguru fields
        filtered_extra = {
            k: v for k, v in extra.items() if not k.startswith("_") and k != "serialized"
        }
        if filtered_extra:
            log_entry["extra"] = filtered_extra

    return json.dumps(log_entry, default=str)


def json_sink(message: Any) -> None:
    """Sink function that outputs JSON formatted logs.

    Args:
        message: The loguru message object.
    """
    record = message.record
    json_str = json_serializer(record)
    sys.stderr.write(json_str + "\n")


def text_format(record: Record) -> str:
    """Format log record as human-readable text.

    Args:
        record: The loguru record to format.

    Returns:
        Formatted string representation.
    """
    correlation_id = get_correlation_id()
    correlation_part = f"[{correlation_id[:8]}] " if correlation_id else ""

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        f"{correlation_part}"
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    file_path: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure the logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: Output format ('json' or 'text').
        file_path: Optional file path for log output.
        rotation: Log rotation size (e.g., "10 MB", "1 day").
        retention: Log retention period (e.g., "7 days", "1 month").
    """
    # Remove default handler
    logger.remove()

    # Add appropriate handler based on format type
    if format_type == "json":
        logger.add(
            json_sink,
            level=level,
            format="{message}",
            colorize=False,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=text_format,
            colorize=True,
        )

    # Add file handler if specified
    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            logger.add(
                str(file_path),
                level=level,
                format="{message}",
                rotation=rotation,
                retention=retention,
                serialize=True,
            )
        else:
            logger.add(
                str(file_path),
                level=level,
                format=text_format,
                rotation=rotation,
                retention=retention,
            )


def get_logger(name: str | None = None) -> Any:
    """Get a logger instance with optional name binding.

    Args:
        name: Optional name to bind to the logger.

    Returns:
        Logger instance bound with the specified name.
    """
    if name:
        return logger.bind(name=name)
    return logger


class LogContext:
    """Context manager for scoped logging with correlation ID and context."""

    def __init__(
        self,
        correlation_id: str | None = None,
        **context: Any,
    ) -> None:
        """Initialize logging context.

        Args:
            correlation_id: Optional correlation ID. Generated if not provided.
            **context: Additional context key-value pairs.
        """
        self._correlation_id = correlation_id
        self._context = context
        self._previous_correlation_id: str | None = None
        self._previous_context: dict[str, Any] = {}

    def __enter__(self) -> LogContext:
        """Enter the logging context."""
        # Save previous state
        self._previous_correlation_id = get_correlation_id()
        self._previous_context = get_request_context()

        # Set new state
        set_correlation_id(self._correlation_id)
        set_request_context(self._context)

        return self

    def __exit__(self, *args: object) -> None:
        """Exit the logging context and restore previous state."""
        # Restore previous state
        if self._previous_correlation_id:
            correlation_id_var.set(self._previous_correlation_id)
        else:
            clear_correlation_id()

        if self._previous_context:
            request_context_var.set(self._previous_context)
        else:
            clear_request_context()

    @property
    def correlation_id(self) -> str | None:
        """Get the correlation ID for this context."""
        return get_correlation_id()


def log_with_context(
    level: str,
    message: str,
    **extra: Any,
) -> None:
    """Log a message with current context and extra fields.

    Args:
        level: Log level.
        message: Log message.
        **extra: Additional fields to include in the log.
    """
    log_func: Callable[..., None] = getattr(logger, level.lower())
    log_func(message, **extra)


# Note: Call configure_logging() in your application startup
# to initialize the logging system. This allows library consumers
# to configure logging before the module is imported elsewhere.
