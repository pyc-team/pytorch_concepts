"""Compatibility helpers for optional Matplotlib style integrations."""


def ensure_style_core_alias() -> None:
    """Expose ``matplotlib.style.core`` when newer Matplotlib omits it.

    Some optional packages imported through TorchMetrics still access the legacy
    alias during import. Keeping this tiny shim before Lightning/TorchMetrics
    imports avoids environment-specific crashes without changing plotting code.
    """
    try:
        import matplotlib.style as mpl_style
    except Exception:
        return
    if not hasattr(mpl_style, "core"):
        mpl_style.core = mpl_style
