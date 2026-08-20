from typing import Union
# Starlette removed fastapi.middleware.wsgi.WSGIMiddleware; a2wsgi provides the
# ASGI->WSGI bridge these Dash/Flask sub-apps are mounted through.
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from rsm.rsm import create_rsm_app
from tb.bandstructure_dash import create_tb_app
from cohp.TBmodel import COHPDashApp
# `/app/gliquid` is DATA ONLY -- static assets, the params workbooks and the read-only
# SQLite store -- served at the public /gliquid/ URL by the mount below. /app is the
# container WORKDIR and therefore first on sys.path, so that directory sits directly in
# the way of the pip-installed `gliquid` package. It is harmless ONLY because it has no
# `__init__.py`: without one it is a mere namespace portion, the import scan continues
# past it, and `import gliquid` resolves to site-packages. Adding an `__init__.py` back
# would make it a regular package and silently shadow the real one -- the original bug.
# For the same reason nothing under it may be importable as `gliquid.<anything>`, which
# is why the Dash app is a top-level module beside it rather than `gliquid/scripts/`.
from gliquid_app import create_gliqtern_app

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/gliquid")
def redirect_to_gliq_im():
    return RedirectResponse(url="/gliquid/interactive-matrix.html")

@app.get("/cogito")
def redirect_to_test_web():
    return RedirectResponse(url="/cogito/bond_plots.html")

@app.get("/vr")
def redirect_to_vr_menu():
    return RedirectResponse(url="/vr/menu.html")

@app.get("/vr/ternary")
def redirect_to_ternary_phase_diagram():
    return RedirectResponse(url="/vr/ternary/index.html")

@app.get("/vr/fermi")
def redirect_to_fermi_surface():
    return RedirectResponse(url="/vr/fermi/index.html")

@app.get("/vr/tomography")
def redirect_to_tomography():
    return RedirectResponse(url="/vr/tomography/index.html")

@app.get("/ternary")
def redirect_to_ternary_phase_diagram():
    return RedirectResponse(url="/vr/ternary/index.html")

@app.get("/fermi")
def redirect_to_fermi_surface():
    return RedirectResponse(url="/vr/fermi/index.html")

@app.get("/tomography")
def redirect_to_tomography():
    return RedirectResponse(url="/vr/tomography/index.html")


dash_app_rsm = create_rsm_app(requests_pathname_prefix="/rsm/")
dash_app_tb = create_tb_app(requests_pathname_prefix="/tb/")
dash_app_cohp = COHPDashApp().create_cohp_dashapp(requests_pathname_prefix="/cogito-cohp/")
dash_app_gliqtern = create_gliqtern_app(requests_pathname_prefix="/gliquid/ternary-interpolation/")

app.mount("/gliquid/ternary-interpolation/", WSGIMiddleware(dash_app_gliqtern.server))
app.mount("/gliquid/", StaticFiles(directory="gliquid"))
app.mount("/cogito/", StaticFiles(directory="cogito"))
app.mount("/cogito-cohp", WSGIMiddleware(dash_app_cohp.server))
app.mount("/rsm", WSGIMiddleware(dash_app_rsm.server))
app.mount("/tb", WSGIMiddleware(dash_app_tb.server))
app.mount("/vr", StaticFiles(directory="vr"))

if __name__ == "__main__":
    app.run()
