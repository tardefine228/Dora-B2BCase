"""The form must persist an actual review, not just show a fake toast."""
import sqlite3

from app.core.config import settings
from app.main import app
from fastapi.testclient import TestClient


def test_feedback_persists_and_rejects_missing_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'job_store', 'sqlite')
    monkeypatch.setattr(settings, 'job_store_path', str(tmp_path / 'jobs.sqlite'))
    with TestClient(app) as client:
        payload = {
            'query': 'Чайник электрический',
            'predicted_code': '27.51.24.110',
            'verdict': 'incorrect',
            'corrected_code': '27.51.24',
        }
        response = client.post('/feedback', json=payload)
        assert response.status_code == 201, response.text
        assert response.json()['saved'] is True
        with sqlite3.connect(tmp_path / 'classification_feedback.sqlite') as conn:
            row = conn.execute(
                'SELECT query, verdict, corrected_code FROM classification_feedback'
            ).fetchone()
            assert row == ('Чайник электрический', 'incorrect', '27.51.24')
        response = client.post('/feedback', json={**payload, 'corrected_code': ''})
        assert response.status_code == 422
