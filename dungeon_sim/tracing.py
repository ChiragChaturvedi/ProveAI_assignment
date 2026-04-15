from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

_TRACING_CONFIGURED = False


def configure_tracing(export_console: bool = False) -> None:
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": "dungeon-sim"}))
    if export_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True


def get_tracer(name: str = "dungeon_sim"):
    configure_tracing()
    return trace.get_tracer(name)


def _normalize_attribute_value(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            item_value = _normalize_attribute_value(item)
            if item_value is not None:
                normalized.append(item_value)
        return normalized
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (bool, int, float, str)):
        return getattr(value, "value")
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), sort_keys=True)
    return str(value)


def safe_set_attributes(span: Span, **attributes: object) -> None:
    for key, value in attributes.items():
        normalized = _normalize_attribute_value(value)
        if normalized is None:
            continue
        span.set_attribute(key, normalized)


def add_span_event(span: Span, name: str, **attributes: object) -> None:
    event_attributes: dict[str, Any] = {}
    for key, value in attributes.items():
        normalized = _normalize_attribute_value(value)
        if normalized is None:
            continue
        event_attributes[key] = normalized
    span.add_event(name, event_attributes)


def set_span_error(span: Span, exc: BaseException, **attributes: object) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    safe_set_attributes(span, error_type=type(exc).__name__, error_message=str(exc), **attributes)


def start_detached_span(name: str, parent_span: Span | None = None, **attributes: object) -> Span:
    tracer = get_tracer()
    context = trace.set_span_in_context(parent_span) if parent_span is not None else None
    span = tracer.start_span(name, context=context)
    safe_set_attributes(span, **attributes)
    return span


def end_span(span: Span | None, **attributes: object) -> None:
    if span is None:
        return
    safe_set_attributes(span, **attributes)
    span.end()


@contextmanager
def node_span(name: str, parent_span: Span | None = None, **attributes: object) -> Iterator[Span]:
    tracer = get_tracer()
    context = trace.set_span_in_context(parent_span) if parent_span is not None else None
    with tracer.start_as_current_span(name, context=context) as span:
        safe_set_attributes(span, **attributes)
        try:
            yield span
        except Exception as exc:
            set_span_error(span, exc)
            raise
