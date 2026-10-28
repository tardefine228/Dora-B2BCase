"""Shared OKPD2 syntax for catalogue uploads and exact code lookup.

Official OKPD2 hierarchy (OK 034-2014), one fixed digit width per level:
``XX | XX.X | XX.XX | XX.XX.X | XX.XX.XX | XX.XX.XX.XXX``.

The final category level is always three digits, so a 2-digit tail
(``22.29.29.19``) and a 3-digit group at an intermediate level
(``22.29.190``) are rejected, not just garbage.

Both the catalogue validator (``app.services.catalog_io``) and the exact-code
lookup in the classifier use this single pattern, so a code accepted on upload
is also the code the classifier recognises when the user types it.
"""

import re

OKPD2_PATTERN = r"^\d{2}(?:\.\d(?:\d(?:\.\d(?:\d(?:\.\d{3})?)?)?)?)?$"
OKPD2_RE = re.compile(OKPD2_PATTERN)
