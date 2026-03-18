.PHONY: install install-all lint typecheck test docs clean

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

lint:
	ruff check src/ projects/ tests/
	ruff format --check src/ projects/ tests/

format:
	ruff format src/ projects/ tests/

typecheck:
	mypy src/risk_analyst/ --ignore-missing-imports

test:
	pytest --tb=short -q

test-cov:
	pytest --cov=risk_analyst --cov-report=html --tb=short

docs:
	@echo "Compile LaTeX docs (requires texlive)"
	@for f in $$(find docs -name "*.tex" ! -name "template.tex"); do \
		echo "Compiling $$f"; \
		latexmk -pdf -cd "$$f"; \
	done

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
