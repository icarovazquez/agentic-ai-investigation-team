"""
Importing this package registers every connector adapter. Add a new
vendor connector as its own module here, following nika.py or
pagerduty.py as a template, then import it below so it self-registers
on package import -- nothing else in the codebase needs to change.
"""

from . import nika, pagerduty  # noqa: F401
