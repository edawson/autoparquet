"""Centralised logging setup for the autoparquet library.

All library modules obtain their logger by calling ``get_logger(__name__)``.
Passing ``__name__`` is sufficient because the package structure already
namespaces every module under ``autoparquet.*``.

The root ``autoparquet`` logger is given a :class:`logging.NullHandler` here
so the library stays silent by default when used as a dependency.  Applications
and the CLI are responsible for attaching their own handlers and calling
:func:`logging.basicConfig` if they want output.
"""

import logging

_ROOT = "autoparquet"

# Silence the library by default.  Applications decide what to show.
logging.getLogger(_ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the autoparquet namespace.

    Args:
        name: Typically ``__name__`` from the calling module, which will
              already be namespaced as ``autoparquet.<module>``.

    Returns:
        A :class:`logging.Logger` instance for the given name.
    """
    return logging.getLogger(name)
