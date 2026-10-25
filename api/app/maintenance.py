"""Retention for the two logs that grow on every request.

`classification_log` gets one row per classified product and `classification_feedback`
one per specialist verdict. Batch jobs were already expired (`JOB_TTL_HOURS`), but
these two grew without a bound: a working deployment would accumulate millions of
rows, and the only way to trim them was an SQL console. The audit asked for a
retention policy (BE-5); this module is it.

Policy, deliberately different for the two tables:

* **audit log** — a copy of confidential procurement descriptions, useful for a
  quarter at most. Default ``AUDIT_RETENTION_DAYS=90``.
* **feedback** — training data for the reranker (ML-4). Deleting it would delete
  the only real-traffic signal we have, so it is kept forever unless an operator
  sets ``FEEDBACK_RETENTION_DAYS`` explicitly (0 = keep).

The purge runs once at startup (so a long-running volume is trimmed on deploy)
and then every ``MAINTENANCE_INTERVAL_HOURS``. `POST /admin/maintenance/purge`
runs it on demand and reports how many rows went.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger("dora.maintenance")

# Table names are constants, never user input: they are interpolated into DELETE.
AUDIT_TABLE = "classification_log"
AUDIT_COLUMN = "ts"
FEEDBACK_TABLE = "classification_feedback"
FEEDBACK_COLUMN = "created_at"


@dataclass(frozen=True)
class PurgeReport:
    """What a purge did (or why it did nothing)."""

    audit_deleted: int = 0
    feedback_deleted: int = 0
    audit_days: int = 0
    feedback_days: int = 0
    skipped: str | None = None
    dry_run: bool = False

    @property
    def deleted(self) -> int:
        return self.audit_deleted + self.feedback_deleted

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "audit_deleted": self.audit_deleted,
            "feedback_deleted": self.feedback_deleted,
            "audit_retention_days": self.audit_days,
            "feedback_retention_days": self.feedback_days,
            "skipped": self.skipped,
            "dry_run": self.dry_run,
        }


_KNOWN_COLUMNS = {
    (AUDIT_TABLE, AUDIT_COLUMN),
    (FEEDBACK_TABLE, FEEDBACK_COLUMN),
}


def _checked(table: str, column: str, days: int) -> None:
    if (table, column) not in _KNOWN_COLUMNS:
        raise ValueError(f"Неизвестная таблица для очистки: {table}.{column}")
    if days < 1:
        raise ValueError("Глубина хранения должна быть не меньше одного дня")


def delete_older_than_sql(table: str, column: str, days: int) -> tuple[str, tuple[int]]:
    """``DELETE`` for rows older than ``days``, parameterised on the interval only.

    Kept separate from the connection so the SQL can be asserted without a
    database: the table and column are constants of this module, and the interval
    is the sole parameter.
    """
    _checked(table, column, days)
    return (
        f"DELETE FROM {table} WHERE {column} < now() - make_interval(days => %s)",
        (int(days),),
    )


def count_older_than_sql(table: str, column: str, days: int) -> tuple[str, tuple[int]]:
    """``SELECT count(*)`` counterpart, for ``dry_run``."""
    _checked(table, column, days)
    return (
        f"SELECT count(*) FROM {table} WHERE {column} < now() - make_interval(days => %s)",
        (int(days),),
    )


def purge_expired(*, dry_run: bool = False) -> PurgeReport:
    """Apply the configured retention. A no-op on non-PostgreSQL backends.

    ``dry_run`` counts what would be deleted and changes nothing — the operator
    can check a retention setting before committing to it.
    """
    audit_days = settings.audit_retention_days
    feedback_days = settings.feedback_retention_days
    if settings.job_store != "postgres":
        # sqlite/memory stores keep neither log (the audit log is PostgreSQL-only).
        return PurgeReport(
            audit_days=audit_days,
            feedback_days=feedback_days,
            skipped="not postgres",
            dry_run=dry_run,
        )
    if audit_days < 1 and feedback_days < 1:
        return PurgeReport(
            audit_days=audit_days,
            feedback_days=feedback_days,
            skipped="retention disabled",
            dry_run=dry_run,
        )

    import psycopg

    audit_deleted = feedback_deleted = 0
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            if audit_days >= 1:
                sql, params = (
                    count_older_than_sql(AUDIT_TABLE, AUDIT_COLUMN, audit_days)
                    if dry_run
                    else delete_older_than_sql(AUDIT_TABLE, AUDIT_COLUMN, audit_days)
                )
                cur.execute(sql, params)
                row = cur.fetchone() if dry_run else None
                audit_deleted = (row[0] if row else cur.rowcount) or 0
            if feedback_days >= 1:
                sql, params = (
                    count_older_than_sql(FEEDBACK_TABLE, FEEDBACK_COLUMN, feedback_days)
                    if dry_run
                    else delete_older_than_sql(
                        FEEDBACK_TABLE, FEEDBACK_COLUMN, feedback_days
                    )
                )
                cur.execute(sql, params)
                row = cur.fetchone() if dry_run else None
                feedback_deleted = (row[0] if row else cur.rowcount) or 0
    report = PurgeReport(
        audit_deleted=audit_deleted,
        feedback_deleted=feedback_deleted,
        audit_days=audit_days,
        feedback_days=feedback_days,
        dry_run=dry_run,
    )
    if report.deleted:
        logger.info(
            "Очистка по сроку хранения: журнал запросов — %d строк (старше %d дн.), "
            "вердикты — %d строк (старше %d дн.)",
            audit_deleted,
            audit_days,
            feedback_deleted,
            feedback_days,
        )
    return report
