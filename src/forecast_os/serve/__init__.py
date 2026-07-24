"""REST serving layer for forecast-os.

Run ``forecast-os-serve`` (requires ``pip install "forecast-os[serve]"``) to
expose the engine over HTTP: discover models and schema mappings, preview how
raw records shape into the ``(unique_id, ds, y)`` panel, then forecast,
compare models, or score quota attainment — the same operations as the MCP
server, reusing its pure tool functions verbatim. See
:mod:`forecast_os.serve.app` for the routes and the :func:`~forecast_os.serve.app.create_app`
factory.
"""
