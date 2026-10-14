from pathlib import Path
from typing import Union
# The ASGI->WSGI bridge the Dash/Flask sub-apps are mounted through.
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from gliqviz.matrix_assets_store import CACHE_CONTROL, MatrixAssetStore
from rsm.rsm import create_rsm_app
from tb.bandstructure_dash import create_tb_app
from cohp.TBmodel import COHPDashApp
from gliqviz.gliquid_app import create_gliqtern_app

app = FastAPI()
# Gzips every response over 1 kB for clients that accept it, mounted Dash apps included.
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=4)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/gliquid")
def redirect_to_gliq_im():
    return RedirectResponse(url="/gliquid/interactive-matrix.html")

@app.get("/maxwell")
def redirect_to_maxwell():
    return RedirectResponse(url="/maxwell/Maxwell_Surface_Viewer.html")

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

# Serves the matrix .webp out of the ZIP shards; None when there are no shards.
# Register it before the StaticFiles mount below: Starlette takes the first match.
_MATRIX_STORE = MatrixAssetStore.load(
    Path(__file__).resolve().parent / "gliqviz" / "matrix_shards")

if _MATRIX_STORE is not None:
    @app.get("/gliquid/matrix_assets/{name}")
    def get_matrix_asset(name: str, request: Request):
        # A sync def, so Starlette runs the blocking zip read in its threadpool.
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
app.mount("/maxwell/", StaticFiles(directory="maxwell"))

if __name__ == "__main__":
    app.run()
