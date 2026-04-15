from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

_TRACING_CONFIGURED = False


def configure_tracing() -> None:
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": "dungeon-sim"}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True


def get_tracer(name: str = "dungeon_sim"):
    configure_tracing()
    return trace.get_tracer(name)


@contextmanager
def node_span(name: str, **attributes: object) -> Iterator[object]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span

