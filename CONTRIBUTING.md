# Contributing to DTS (Dialogue Tree Search)

Thank you for your interest in contributing to DTS! This document outlines the guidelines and standards for contributing to this project.

## Getting Started

### Prerequisites

- Python 3.11+
- `uv` package manager
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/MVPandey/DTS.git
cd DTS

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies (including dev dependencies)
uv pip install -e .
uv pip install pre-commit mypy

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) with **mandatory issue linking**. Every commit must reference the GitHub issue it addresses.

### Format

```
<type>(<scope>): <short summary>

<body with issue reference>

[optional footer]
```

### Types

| Type       | Description                                          |
|------------|------------------------------------------------------|
| `feat`     | A new feature                                        |
| `fix`      | A bug fix                                            |
| `docs`     | Documentation only changes                           |
| `style`    | Code style changes (formatting, whitespace, etc.)    |
| `refactor` | Code change that neither fixes a bug nor adds feature|
| `perf`     | Performance improvement                              |
| `test`     | Adding or updating tests                             |
| `build`    | Build system or dependency changes                   |
| `ci`       | CI/CD configuration changes                          |
| `chore`    | Other changes that don't modify src or test files    |
| `revert`   | Reverting a previous commit                          |

### Scopes (Optional)

Use a scope to indicate what part of the codebase is affected:

| Scope      | Description                                          |
|------------|------------------------------------------------------|
| `api`      | API server and endpoints                             |
| `engine`   | DTS engine core logic                                |
| `llm`      | LLM client and integrations                          |
| `ui`       | Frontend/UI changes                                  |
| `config`   | Configuration changes                                |
| `deps`     | Dependency updates                                   |

### Examples

```bash
# Feature addition (linking issue in body)
feat(engine): add confidence-based branching

Implements confidence scoring to reduce token usage by pruning
low-confidence branches early.

Closes #1

# Bug fix
fix(api): resolve race condition in concurrent LLM calls

Closes #42

# Documentation (scope optional for docs)
docs: update API documentation for new endpoints

Closes #8

# Refactoring
refactor(engine): extract scoring logic into separate module

Part of backend cleanup effort.

Closes #8

# Multiple issues
feat(engine): add prompt optimization mode

Implements core prompt fitting functionality.

Closes #5
Related: #7, #3

# Quick fix (no scope)
fix: correct typo in error message

Closes #99
```

### Rules

1. **Always link an issue** - Use `Closes #<issue>` or `Fixes #<issue>` in the commit body
2. **Use imperative mood** - "add feature" not "added feature"
3. **Keep summary under 72 characters**
4. **No period at the end** of the summary line
5. **Separate body from summary** with a blank line
6. **Scope is optional** but encouraged for clarity

## Code Standards

### Formatting & Linting

We use `ruff` for both linting and formatting:

```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Format code
ruff format .
```

### Type Hints

All code must be fully type-hinted. We use `mypy` for type checking:

```bash
# Run type checker
mypy backend/
```

**Guidelines:**
- Use Python 3.10+ style hints (e.g., `list[str]` not `List[str]`)
- All function parameters must have type hints
- All function return types must be specified
- Use `|` for unions (e.g., `str | None` not `Optional[str]`)

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=backend
```

**Note:** Unit tests are currently being implemented (see Issue #11).

## Pre-commit Hooks

Pre-commit hooks run automatically on every commit:

1. **ruff (lint)** - Checks for code issues
2. **ruff (format)** - Ensures consistent formatting
3. **mypy** - Validates type hints
4. **conventional-pre-commit** - Validates commit message format
5. **General checks** - Trailing whitespace, YAML/JSON validity, etc.

### Bypassing Hooks (Emergency Only)

```bash
# Skip all hooks (use sparingly!)
git commit --no-verify -m "feat(#1): emergency fix"
```

## Pull Request Process

1. **Create an issue first** (if one doesn't exist)
2. **Branch from `main`** using the naming convention:
   - `feat/#<issue>-short-description`
   - `fix/#<issue>-short-description`
   - `refactor/#<issue>-short-description`
3. **Make your changes** with proper commits
4. **Ensure all checks pass**:
   ```bash
   ruff check .
   ruff format . --check
   mypy backend/
   pytest
   ```
5. **Open a PR** linking to the issue
6. **Request review** and address feedback

### PR Title Format

Use the same conventional commit format (issue can be in title or description):
```
feat(engine): add confidence-based branching (#1)
```
or simply:
```
feat(engine): add confidence-based branching
```
The PR description should reference the issue with `Closes #<issue-number>`.

## Project Structure

```
MCTS/
├── backend/
│   ├── api/          # FastAPI server and endpoints
│   ├── core/
│   │   └── dts/      # Core DTS engine and components
│   └── llm/          # LLM client abstraction
├── frontend/         # React/Next.js frontend
├── tests/            # Unit and integration tests
└── media/            # Static assets and documentation
```

## Questions?

- Open a [GitHub Discussion](https://github.com/MVPandey/DTS/discussions)
- Check existing [Issues](https://github.com/MVPandey/DTS/issues)

Thank you for contributing! 🎉
