# 🤝 Contributing to Promptolution

Thank you for your interest in contributing! Here's how to get started.

## Workflow

Open an issue → create a branch → PR → CI → review (by owner) → merge (by owner) → release (by owner)

Branch naming: `feature/...`, `fix/...`, `chore/...`, `refactor/...`.

## Code Quality

Please ensure to use pre-commit, which assists with keeping the code quality high:

```
pre-commit install
pre-commit run --all-files
```

## Tests

We encourage every contributor to also write tests that automatically check if the implementation works as expected:

```
poetry run python -m coverage run -m pytest
poetry run python -m coverage report -i
```