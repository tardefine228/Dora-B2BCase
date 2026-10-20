"""Test-wide defaults.

Background jobs must not leak into the repository, so the whole suite runs on
the in-memory store; tests that need persistence switch to ``sqlite`` explicitly
(see ``test_job_store.py``).

Rate limiting is off by default as well: most tests hammer the same
``testclient`` address. ``test_ratelimit.py`` turns it back on per test.
"""

from app.core.config import settings

settings.job_store = "memory"
# Unit tests exercise the lexical path; vector search has its own tests
# (test_vector_search.py) that enable it with a small deterministic encoder.
settings.vector_search = False
settings.rate_limit_enabled = False
