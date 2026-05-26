# `nrcatalogtools.registry`

Lightweight catalog plugin registry.

Catalogs register themselves with `@register_catalog("TAG")` and are looked up with
`get_catalog("TAG")`.  The built-in catalogs (RIT, SXS, MAYA) are registered automatically
at import time.

---

::: nrcatalogtools.registry.register_catalog

::: nrcatalogtools.registry.get_catalog

::: nrcatalogtools.registry.list_catalogs
