import os
import zipfile

for zip_path in ["gliquid/matrix_assets.zip", "gliquid/binary_cache.zip"]:
    target_dir = zip_path.replace('.zip', '')
    if os.path.exists(zip_path) and not os.path.exists(target_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(target_dir))
    assert os.path.exists(target_dir), f"Required directory {target_dir} does not exist"

from typing import Union
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from rsm.rsm import create_rsm_app
from tb.bandstructure_dash import create_tb_app
from cohp.TBmodel import COHPDashApp            
from gliquid.scripts.interactive_ternary_plotter import create_gliqtern_app

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
def redirect_to_ternary_phase_diagram():
    return RedirectResponse(url="/vr/ternary.html")


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
