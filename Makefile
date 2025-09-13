.PHONY: setup
setup:
	uv venv --python=python3.12
	. .venv/bin/activate && make sync

.PHONY: sync
sync:
	uv pip sync requirements/requirements.txt

.PHONY: update-requirements
update-requirements:
	uv pip compile -U requirements/requirements.in --universal --output-file requirements/requirements.txt

.PHONY: format
format:
	@echo "Running pre-commit"
	uvx pre-commit run --all-files
