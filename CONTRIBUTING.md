# Contributing to NNDS

We welcome contributions! Please follow these guidelines:

## Reporting Issues
- Use the GitHub issue tracker.
- Provide a clear description and steps to reproduce.

## Pull Requests
- Fork the repository and create a new branch.
- Ensure your code passes all tests (`pytest`).
- Format your code with `black` and `isort`.
- Write clear commit messages.

## Development Setup

```bash
pip install -e .[dev]
pre-commit install
```

## Code Style
- We follow PEP 8.
- Use `black` (line length 100) and `isort`.

## Testing
- Write tests for new features.
- Run `pytest` to ensure no regressions.

Thank you for contributing!
