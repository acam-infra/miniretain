"""OCR adapter boundary, provider registry, and document.v1 output contract.

To add a new OCR backend:
1. Subclass ``OCRProvider`` and implement ``parse(self, source: str) -> list[dict]``.
2. Return normalized pages produced by ``ocr.normalize.normalize_pages(...)`` in
   either ``pdf_pt`` (72 DPI) or ``pixel`` (rasterization DPI) coordinate space.
3. Register the backend via ``register_provider("my-backend", MyProvider)`` (and add
   the name to ``contracts/task.spec.schema.json`` ``ocr.provider.enum`` when exposing
   it through the Go HTTP API / YAML config).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ocr.base import (  # noqa: F401
    ALLOWED_COORD_SPACES,
    OCRBackendUnavailableError,
    OCRConfigError,
    OCRContractError,
    OCRError,
    OCRProvider,
    validate_document_pages,
)
from ocr.normalize import normalize_pages


class StubOCRProvider(OCRProvider):
    """Offline stand-in: treats the source text as an already-OCR'd page."""

    name = "stub"

    def parse(self, source: str) -> list[dict]:
        blocks = []
        for para in source.split("\n\n"):
            text = " ".join(line.strip() for line in para.splitlines() if line.strip())
            if text:
                blocks.append({"block_id": f"b{len(blocks):04d}", "kind": "text", "text": text})
        raw_pages = [{"page_index": 0, "width": 595, "height": 842, "blocks": blocks}]
        return normalize_pages(raw_pages, coord_space="pdf_pt", dpi=72.0)


from ocr.paddle_adapter import PaddleOCRProvider  # noqa: E402,F401
from ocr.paddle_http_adapter import PaddleHTTPOCRProvider  # noqa: E402,F401
from ocr.paddle_cloud_adapter import PaddleCloudOCRProvider  # noqa: E402,F401
from ocr.mineru_cloud_adapter import MinerUCloudOCRProvider  # noqa: E402,F401
from ocr.pymupdf_adapter import AutoOCRProvider, PyMuPDFOCRProvider  # noqa: E402,F401


class _LazyPaddle(PaddleOCRProvider):
    """Registered lazily: heavy models load on first parse(), not at import."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        OCRProvider.__init__(self, config=config)
        self._ocr = None
        self._fitz = None
        self._initialized = False

    def parse(self, source: str) -> list[dict]:
        if not self._initialized:
            super().__init__(config=self.config)
            self._initialized = True
        return super().parse(source)


ProviderFactory = Callable[[dict[str, Any] | None], OCRProvider]

REGISTRY: dict[str, OCRProvider] = {}
_FACTORIES: dict[str, ProviderFactory] = {}


def _to_factory(provider_or_factory: type[OCRProvider] | ProviderFactory | OCRProvider) -> ProviderFactory:
    if isinstance(provider_or_factory, OCRProvider):
        inst = provider_or_factory
        return lambda _cfg: inst
    if isinstance(provider_or_factory, type) and issubclass(provider_or_factory, OCRProvider):
        cls = provider_or_factory
        return lambda cfg: cls(config=cfg)
    if callable(provider_or_factory):
        return provider_or_factory
    raise OCRConfigError(
        f"provider registration expects an OCRProvider subclass, instance, or callable; got {type(provider_or_factory).__name__}"
    )


def register_provider(
    name: str,
    provider_or_factory: type[OCRProvider] | ProviderFactory | OCRProvider,
    *,
    replace: bool = False,
    eager: bool = False,
) -> None:
    """Register an OCR backend by name for lookup via ``get_provider``.

    By default (``eager=False``), provider classes and factory callables are
    stored lazily without calling ``factory(None)`` at registration time, so
    backends that require runtime configuration (such as ``ocr.options.endpoint``)
    in ``__init__`` can be registered at import time without failing or affecting
    other backends. Built-in providers register with ``eager=True`` so
    ``REGISTRY`` retains their default instances at import time (with
    ``paddle-cpu`` still loading heavy models lazily on first ``parse()``).
    """
    if not isinstance(name, str) or not name.strip():
        raise OCRConfigError("provider name must be a non-empty string")
    key = name.strip()
    if (key in _FACTORIES or key in REGISTRY) and not replace:
        raise OCRConfigError(f"OCR provider {key!r} is already registered (pass replace=True to override)")
    if isinstance(provider_or_factory, OCRProvider):
        inst = provider_or_factory
        _FACTORIES[key] = lambda _cfg, _inst=inst: _inst
        REGISTRY[key] = inst
    else:
        factory = _to_factory(provider_or_factory)
        _FACTORIES[key] = factory
        if eager:
            REGISTRY[key] = factory(None)
        else:
            REGISTRY.pop(key, None)


def unregister_provider(name: str) -> None:
    """Remove a registered OCR provider (primarily for tests)."""
    _FACTORIES.pop(name, None)
    REGISTRY.pop(name, None)


def list_providers() -> list[str]:
    """Return sorted names of all registered OCR providers."""
    return sorted(set(REGISTRY) | set(_FACTORIES))


def get_provider(name: str, config: dict[str, Any] | None = None) -> OCRProvider:
    """Resolve a registered OCR provider, optionally configured with ``config``."""
    if config is not None and not isinstance(config, dict):
        raise OCRConfigError(f"ocr config must be a dict or None, got {type(config).__name__}")
    if config:
        if name in _FACTORIES:
            return _FACTORIES[name](config)
        if name in REGISTRY:
            return REGISTRY[name]
    else:
        if name in REGISTRY:
            return REGISTRY[name]
        if name in _FACTORIES:
            inst = _FACTORIES[name](None)
            REGISTRY[name] = inst
            return inst
    raise OCRConfigError(f"unknown OCR provider {name!r}; available: {list_providers()}")


register_provider(StubOCRProvider.name, StubOCRProvider, eager=True)
register_provider(PyMuPDFOCRProvider.name, PyMuPDFOCRProvider, eager=True)
register_provider(PaddleHTTPOCRProvider.name, PaddleHTTPOCRProvider, eager=True)
register_provider(PaddleCloudOCRProvider.name, PaddleCloudOCRProvider, eager=True)
register_provider(MinerUCloudOCRProvider.name, MinerUCloudOCRProvider, eager=True)
register_provider(AutoOCRProvider.name, AutoOCRProvider, eager=True)
register_provider(_LazyPaddle.name, _LazyPaddle, eager=True)
