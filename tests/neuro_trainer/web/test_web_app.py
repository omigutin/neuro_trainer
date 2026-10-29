import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neuro_trainer.config_store import ConfigStore
from neuro_trainer.job_manager import JobManager
from neuro_trainer.neuro_trainer import NeuroTrainerApp
from neuro_trainer.presets import PresetRegistry
from neuro_trainer.web.web_app import WebContext, create_app, mount


@pytest.fixture
def ctx(paths):
    app = NeuroTrainerApp(paths, ConfigStore(paths), PresetRegistry(paths.presets_dir))
    return WebContext(trainer_app=app, jobs=JobManager())


@pytest.fixture
def client(ctx):
    return TestClient(create_app(ctx))


def test_page_and_static(client):
    html = client.get("/").text
    assert "Neuro Trainer" in html and "/static/app.js" in html
    assert client.get("/static/app.js").status_code == 200


def test_meta_has_help_for_every_param(client):
    meta = client.get("/api/meta").json()
    params = [p for g in meta["groups"] for p in g["params"]]
    assert params and all(p["help"] for p in params)
    assert {"model_source", "data", "train_mode", "profile"} <= set(meta["help"])
    yolo26 = next(f for f in meta["families"] if f["key"] == "yolo26")
    assert set(yolo26["models"]) == {"classifier", "detector", "segmentation", "pose", "obb"}


def test_state_update_and_errors(client):
    res = client.post("/api/state", json={"model_name": "yolo26m-obb.pt", "epochs": 0}).json()
    assert res["state"]["task"] == "obb" and "epochs" in res["errors"]
    assert client.get("/api/state").json()["state"]["model_name"] == "yolo26m-obb.pt"


def test_profile(client):
    res = client.get("/api/profile", params={"task": "detector", "profile": "low_vram"}).json()
    assert res["values"]["batch"] == 8
    assert client.get("/api/profile", params={"task": "x", "profile": "fast"}).status_code == 400


def test_user_presets(client):
    assert client.post("/api/user-presets/save", json={"name": "a"}).json()["names"] == ["a"]
    assert client.post("/api/user-presets/load", json={"name": "a"}).status_code == 200
    assert client.post("/api/user-presets/load", json={"name": "zz"}).status_code == 400


def test_inspect_errors(client):
    assert client.post("/api/model/inspect", json={"path": "missing.pt"}).status_code == 400


def test_job_flow(client, tmp_path, fake_ultralytics, paths):
    (paths.weights_dir / "yolo26n.pt").write_text("w")
    assert client.post("/api/job/train").status_code == 400  # нет датасета
    client.post("/api/state", json={"model_name": "yolo26n.pt", "data": "d.yaml", "epochs": 3})
    res = client.post("/api/job/train")
    assert res.status_code == 200, res.text
    end = time.time() + 30
    while time.time() < end:
        st = client.get("/api/job/status").json()
        if not st["job"]["active"]:
            break
        time.sleep(0.2)
    assert st["job"]["state"] == "finished", st["log"]
    assert len(st["rows"]) == 3 and st["job"]["run_dir"].endswith("run1")
    assert client.post("/api/job/stop").status_code == 409


def test_mount_under_prefix(paths, ctx):
    host = FastAPI()
    mount(host, "/trainer", ctx)
    c = TestClient(host)
    assert 'data-base="/trainer"' in c.get("/trainer/").text
    assert c.get("/trainer/static/app.css").status_code == 200
    assert c.get("/trainer/api/meta").status_code == 200
