from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_read_server():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "working", "ver": "v1"}


def test_get_languages():
    response = client.get("/langs")
    assert response.status_code == 200
    assert response.json() != None


def test_post_lang_id_french():
    response = client.post(
        "/lang_id/",
        json={
            "text": "Le parole est l\u0027ombre du fait",
        },
    )
    assert response.status_code == 200
    assert response.json() == {'French': '1.00'}


def test_post_lang_id_chinese():
    response = client.post(
        "/lang_id/",
        json={
            "text": "什么是生命的意义",
        },
    )
    assert response.status_code == 200
    assert response.json() == {'Chinese': '1.00'}
