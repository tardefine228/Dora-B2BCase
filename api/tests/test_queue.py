"""Batch dispatch (in-process vs Celery) and the Celery task body."""

import pytest
from app import dispatch
from app.core.config import settings
from app.job_store import create_job, get_job, reset_store
from app.services.classifier import get_classifier, reset_classifier
from app.tasks import run_batch_job
from fastapi import BackgroundTasks, HTTPException


@pytest.fixture(autouse=True)
def fresh_store():
    reset_store()
    reset_classifier()
    yield
    reset_store()
    reset_classifier()


def test_inprocess_dispatch_uses_background_tasks(monkeypatch):
    monkeypatch.setattr(settings, "queue_backend", "inprocess")
    tasks = BackgroundTasks()
    assert dispatch.enqueue_batch("job-1", tasks, lambda _id: None) == "inprocess"
    assert len(tasks.tasks) == 1


def test_celery_dispatch_sends_task_to_the_broker(monkeypatch):
    from app import worker

    sent = []
    monkeypatch.setattr(settings, "queue_backend", "celery")
    monkeypatch.setattr(
        worker.celery_app, "send_task", lambda name, args=None: sent.append((name, args))
    )
    tasks = BackgroundTasks()
    assert dispatch.enqueue_batch("job-2", tasks, lambda _id: None) == "celery"
    assert sent == [(dispatch.BATCH_TASK_NAME, ["job-2"])]
    assert not tasks.tasks  # nothing runs inside the API process


def test_unreachable_broker_fails_the_job_and_returns_503(monkeypatch):
    from app import worker

    def boom(*_args, **_kwargs):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(settings, "queue_backend", "celery")
    monkeypatch.setattr(worker.celery_app, "send_task", boom)
    job = create_job(["чайник"])
    with pytest.raises(HTTPException) as caught:
        dispatch.enqueue_batch(job.job_id, BackgroundTasks(), lambda _id: None)
    assert caught.value.status_code == 503
    assert get_job(job.job_id).status == "failed"


def test_celery_task_processes_a_batch():
    from app.worker import classify_batch

    job = create_job(["Ноутбук Lenovo", "Электрочайники"])
    classify_batch.run(job.job_id)  # same body the worker executes
    done = get_job(job.job_id)
    assert done.status == "completed" and done.completed == 2
    assert [r.code for r in done.results][0] == "26.20.11"


def test_run_batch_job_fails_cleanly_when_classifier_is_unavailable():
    job = create_job(["чайник"])

    def unavailable():
        raise RuntimeError("no classifier")

    run_batch_job(job.job_id, unavailable)
    assert get_job(job.job_id).status == "failed"


def test_get_classifier_rebuilds_when_the_catalog_file_changes(tmp_path, monkeypatch):
    source = settings.data_dir
    catalog = tmp_path / "data.csv"
    catalog.write_text(
        "record_id;appendix;position;name;okpd2\n1;1;1;Чайники электрические;27.51.24.110\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "catalog_path", str(catalog))
    first = get_classifier()
    assert get_classifier() is first
    catalog.write_text(
        "record_id;appendix;position;name;okpd2\n1;1;1;Станки токарные;28.41.2\n",
        encoding="utf-8",
    )
    second = get_classifier()
    assert second is not first
    assert second.classify("станок токарный").code == "28.41.2"
    assert source  # settings untouched by the test
