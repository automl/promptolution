class DummyConfig:
    """Lightweight config stub used across tests."""

    def __init__(self, task_description=None):
        self.applied = False
        self.validated = False
        self.task_description = task_description

    def apply_to(self, obj):
        self.applied = True
        obj.config_applied = True
        obj.applied = True

    def validate(self):
        self.validated = True
