"""Backend-related definitions and routines.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import warnings

try:
    from .. import _sealwatch
except ImportError:  # extension not built, e.g., installed from source without Rust
    _sealwatch = None


class Backend(enum.Enum):
    """Backend implementation to choose from."""

    BACKEND_PYTHON = enum.auto()
    """Python (baseline) implementation."""
    BACKEND_RUST = enum.auto()
    """RUST implementation, if available."""

    def __enter__(self):
        global backend
        self._backend_prev = backend
        backend = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global backend
        backend = self._backend_prev


def rust_module(name: str):
    """Rust implementation of a module, or None if the extension is not available."""
    return getattr(_sealwatch, name) if _sealwatch is not None else None


if _sealwatch is not None:
    backend = Backend.BACKEND_RUST
else:
    warnings.warn('Rust extension of sealwatch not available, using the Python implementations')
    backend = Backend.BACKEND_PYTHON


def get_backend() -> Backend:
    return backend


def set_backend(b: Backend):
    global backend
    backend = b
