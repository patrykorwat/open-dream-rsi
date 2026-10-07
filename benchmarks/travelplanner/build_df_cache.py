#!/usr/bin/env python3
"""Build the pickle frame cache consumed by sandbox_server.Sandbox.

Frames are built by the OFFICIAL TravelPlanner tool classes, exactly as the
server built them before caching (Flights.load_db() overrides the __init__
frame, so flights.data is the post-load_db frame). Rebuild after any DB swap.
Startup-only optimisation — no data fork; see sandbox_server.load_frame.

  TP_PY=.venv-tp/bin/python python3 build_df_cache.py
"""
import contextlib
import io
import os
import pickle
import sys
import types

TP_REPO = os.environ.get("TP_REPO", "/tmp/tp")
sys.path.insert(0, TP_REPO)
sys.modules.setdefault("gradio", types.ModuleType("gradio"))
from tools.accommodations.apis import Accommodations  # noqa: E402
from tools.flights.apis import Flights  # noqa: E402
from tools.attractions.apis import Attractions  # noqa: E402
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix  # noqa: E402
from tools.restaurants.apis import Restaurants  # noqa: E402

DB = os.environ.get("TP_DB", TP_REPO + "/database")
OUT = os.environ.get("TP_DF_CACHE", TP_REPO + "/df_cache")

os.chdir(os.path.join(TP_REPO, "evaluation"))  # GoogleDistanceMatrix cwd-relative path
out = {}
with contextlib.redirect_stdout(io.StringIO()):
    fl = Flights(path=DB + "/flights/clean_Flights_2022.csv")
    fl.load_db()
    out["flights"] = fl.data
    out["accom"] = Accommodations(
        path=DB + "/accommodations/clean_accommodations_2022.csv").data
    out["rest"] = Restaurants(
        path=DB + "/restaurants/clean_restaurant_2022.csv").data
    out["gdm"] = GoogleDistanceMatrix().data
    out["attr"] = Attractions(path=DB + "/attractions/attractions.csv").data

os.makedirs(OUT, exist_ok=True)
for k, v in out.items():
    pickle.dump(v, open(f"{OUT}/{k}.pkl", "wb"))
    print(k, v.shape)
