from pathlib import Path
from typing import Union
# Starlette removed fastapi.middleware.wsgi.WSGIMiddleware; a2wsgi provides the
# ASGI->WSGI bridge these Dash/Flask sub-apps are mounted through.
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from gliqviz.matrix_assets_store import CACHE_CONTROL, MatrixAssetStore
from rsm.rsm import create_rsm_app
from tb.bandstructure_dash import create_tb_app
from cohp.TBmodel import COHPDashApp
from gliqviz.gliquid_app import create_gliqtern_app

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

# Serves the matrix .webp out of the ZIP shards at their original urls. MUST precede the
# StaticFiles mount below -- Starlette takes the first match in registration order. Loads as
# None when no shards are present, falling through to the mount.
_MATRIX_STORE = MatrixAssetStore.load(
    Path(__file__).resolve().parent / "gliqviz" / "matrix_shards")

if _MATRIX_STORE is not None:
    @app.get("/gliquid/matrix_assets/{name}")
    def get_matrix_asset(name: str, request: Request):
        # Sync def: Starlette runs it in the threadpool, so the blocking zip read never stalls
        # the event loop.
        etag = _MATRIX_STORE.etag(name)
        if etag is None:
            return Response(status_code=404)
        headers = {"ETag": etag, "Cache-Control": CACHE_CONTROL}
        if etag in [t.strip() for t in request.headers.get("if-none-match", "").split(",")]:
            return Response(status_code=304, headers=headers)
        return Response(content=_MATRIX_STORE.read(name), media_type="image/webp", headers=headers)

app.mount("/gliquid/", StaticFiles(directory="gliqviz"))
app.mount("/cogito/", StaticFiles(directory="cogito"))
app.mount("/cogito-cohp", WSGIMiddleware(dash_app_cohp.server))
app.mount("/rsm", WSGIMiddleware(dash_app_rsm.server))
app.mount("/tb", WSGIMiddleware(dash_app_tb.server))
app.mount("/vr", StaticFiles(directory="vr"))

if __name__ == "__main__":
    app.run()
