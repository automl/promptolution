"""Lightweight config stub used across tests."""


class DummyConfig:
    """Minimal config object that tracks apply/validate calls."""

    def __init__(self, task_description=None):
        """Initialize the dummy config with an optional task description."""
        self.applied = False
        self.validated = False
        self.task_description = task_description

    def apply_to(self, obj):
        """Mark the target object as having config applied."""
        self.applied = True
        obj.config_applied = True
        obj.applied = True

    def validate(self):
        """Record that validation was executed."""
        self.validated = True
