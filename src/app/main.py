from typing import Union
import uvicorn
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI
from app.rsm.rsm import create_rsm_app
from app.tb.bandstructure_dash import create_tb_app
from app.tb.bandstructure_dash import Widget
from app.cohp.TBmodel import COHPDashApp
import flask
from dash import Dash
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

dash_app_rsm = create_rsm_app(requests_pathname_prefix="/rsm/")
dash_app_tb = create_tb_app(requests_pathname_prefix="/tb/")
dash_app_cohp = COHPDashApp().create_cohp_dashapp(requests_pathname_prefix="/cogito-cohp/")

app.mount("/rsm", WSGIMiddleware(dash_app_rsm.server))
app.mount("/tb", WSGIMiddleware(dash_app_tb.server))
app.mount("/cogito-cohp", WSGIMiddleware(dash_app_cohp.server))

if __name__ == "__main__":
    app.run()
