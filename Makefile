.PHONY: help

help:
	@echo "Usage:"
	@echo "  make check              CI: Lint the code"
	@echo "  make format             CI: Format the code"
	@echo "  make type               CI: Check typing"
	@echo "  make build              Build the package wheel before publishing to Pypi"
	@echo "  make publish            Publish package to Pypi"

check:
	uv run ruff check $$(git diff --name-only --cached -- '*.py')

format:
	uv run ruff format $$(git diff --name-only --cached -- '*.py')

type:
	uv run ty check $$(git diff --name-only --cached -- '*.py')

build:
	uv build

publish:
	uv publish

commit:
	uv run pre-commit