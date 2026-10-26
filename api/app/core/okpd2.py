"""Shared OKPD2 syntax for catalogue uploads and exact code lookup."""

import re

# Class, subclass, group, subgroup, type and nine-digit category (e.g. 22.29.29.190).
OKPD2_RE = re.compile(r"[0-9]{2}(?:\.[0-9]{1,2}(?:\.[0-9]{1,2}(?:\.[0-9]{3})?)?)?")
