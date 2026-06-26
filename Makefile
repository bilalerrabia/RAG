install:
	uv sync

run:
	uv run python3 -m src pipeline

debug:
	python3 -m pdb src/__main__.py

clean:
	rm -rf __pycache__ .mypy_cache
	rm -rf src/__pycache__

lint:
	-flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	-flake8 src
	mypy src --strict
