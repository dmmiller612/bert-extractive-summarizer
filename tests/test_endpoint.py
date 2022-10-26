import pytest
from application import app as test_app


@pytest.fixture()
def app():
    test_app.config.update({
        "TESTING": True,
    })
    yield test_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_post_endpoint(client):
    input_path = 'tests/fixtures/1152.clean.txt'
    with open(input_path) as input_file:
        text = input_file.read()
        response = client.post("/summarize?num_sentences=1", data=text)
        assert response.status_code == 200
        assert "summary" in response.json


def test_post_endpoint_skip_first(client):
    input_path = 'tests/fixtures/1152.clean.txt'
    with open(input_path) as input_file:
        text = input_file.read()
        response = client.post("/summarize?num_sentences=2&use_first=False", data=text)
        assert response.status_code == 200
        assert response.json.get("summary").startswith("First, participants walked an outbound path")


def test_post_endpoint_use_first(client):
    input_path = 'tests/fixtures/1152.clean.txt'
    with open(input_path) as input_file:
        text = input_file.read()
        response = client.post("/summarize?num_sentences=2&use_first=True", data=text)
        assert response.status_code == 200
        assert response.json.get("summary").startswith("Our ability to return to the start of a route")


def test_get_endpoint(client):
    response = client.get("/summarize")
    assert response.status_code == 405

