.PHONY: install test check

install:
	python -m pip install -e .

test:
	python -m unittest discover -s tests -v

check: test
	git diff --check
