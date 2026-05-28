"""Catalog plugin registry.

Provides a lightweight decorator + lookup mechanism so that new catalogs
can be registered without editing ``__init__.py`` or any core module.

Built-in registration
---------------------
``RITCatalog``, ``SXSCatalog``, and ``MayaCatalog`` are all registered
automatically when the ``nrcatalogtools`` package is imported.

Third-party registration
------------------------
A downstream package (or a user in an interactive session) can register
an additional catalog at runtime::

    from nrcatalogtools.registry import register_catalog
    from nrcatalogtools.catalog import CatalogBase

    @register_catalog("LVCNR")
    class LVCNRCatalog(CatalogBase):
        CATALOG_TYPE = "LVCNR"
        ...

Lookup
------
::

    from nrcatalogtools.registry import get_catalog
    cls = get_catalog("RIT")   # → RITCatalog
    obj = cls.load()
"""

from __future__ import annotations

from typing import Callable, Type

_REGISTRY: dict[str, type] = {}


def register_catalog(tag: str) -> Callable[[Type], Type]:
    """Class decorator that registers a catalog under *tag*.

    Parameters
    ----------
    tag : str
        Short uppercase identifier (e.g. ``"RIT"``).  Must be unique
        within the registry.

    Returns
    -------
    callable
        A class decorator; the class itself is returned unchanged so the
        decorator can be stacked with other decorators.

    Raises
    ------
    ValueError
        If *tag* is already registered, to prevent silent overwrites.

    Examples
    --------
    >>> @register_catalog("LVCNR")
    ... class LVCNRCatalog(CatalogBase):
    ...     CATALOG_TYPE = "LVCNR"
    """

    def decorator(cls):
        if tag in _REGISTRY and _REGISTRY[tag].__name__ != cls.__name__:
            raise ValueError(
                f"Catalog tag '{tag}' is already registered "
                f"(by {_REGISTRY[tag].__qualname__}). "
                "Use a different tag or unregister the existing entry first."
            )
        _REGISTRY[tag] = cls
        return cls

    return decorator


def get_catalog(tag: str) -> Type:
    """Return the catalog class registered under *tag*.

    Parameters
    ----------
    tag : str
        Short uppercase identifier (e.g. ``"RIT"``).

    Returns
    -------
    type
        The registered catalog class.

    Raises
    ------
    KeyError
        If *tag* is not in the registry.

    Examples
    --------
    >>> cls = get_catalog("SXS")
    >>> catalog = cls.load()
    """
    if tag not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"No catalog registered under tag '{tag}'. " f"Known tags: {known}."
        )
    return _REGISTRY[tag]


def list_catalogs() -> set:
    """Return the set of all registered catalog tags.

    Returns
    -------
    set[str]
        Copy of the current tag set; modifying it has no effect on the registry.

    Examples
    --------
    >>> list_catalogs()
    {'MAYA', 'RIT', 'SXS'}
    """
    return set(_REGISTRY)
