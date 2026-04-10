"""Shared test configuration for modeldiff."""

import sys

# Register plugin only when entry-point auto-discovery hasn't loaded it
if "modeldiff.plugin" not in sys.modules:
    pytest_plugins = ["modeldiff.plugin"]
