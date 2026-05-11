"""Endpoint tests using FastAPI TestClient with mocked model."""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

class FakeTokenized(dict):
    """Dict that also has .to() so it matches what HF tokenizers return."""
    def to(self, device):
        return self

@pytest.fixture
def client():
    from serving.api import main as api

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 0
    fake_input_ids = MagicMock()
    fake_input_ids.shape = (1, 5)
    fake_tok.return_value = FakeTokenized(input_ids=fake_input_ids)
    fake_tok.decode.return_value = "add ban check to login\n"

    fake_model = MagicMock()
    fake_generated = MagicMock()
    fake_generated.__getitem__.return_value = list(range(10))
    fake_model.generate.return_value = fake_generated

    api.state.update(tok=fake_tok, model=fake_model, device="cpu")
    yield TestClient(api.app)
    api.state.clear()

def test_health_no_model():
    from serving.api import main as api
    api.state.clear()
    c = TestClient(api.app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False

def test_health_with_model(client):
    r = client.get("/health")
    assert r.json()["model_loaded"] is True

def test_generate_returns_message(client):
    r = client.post("/generate", json={"diff": "# MODIFY x.py\n+ print('hi')"})
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    assert body["cached"] is False

def test_generate_rejects_short_diff(client):
    r = client.post("/generate", json={"diff": "x"})
    assert r.status_code == 422

def test_generate_rejects_missing_field(client):
    r = client.post("/generate", json={})
    assert r.status_code == 422
