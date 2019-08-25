.PHONY: help meta install clean test coverage


TARGET ?= test


.DEFAULT: help
help:
	@echo "make meta"
	@echo "       update version number and meta data"
	@echo "make install"
	@echo "       install reinforcement and dependencies in currently active environment"
	@echo "make clean"
	@echo "       clean all python build/compiliation files and directories"
	@echo "make test"
	@echo "       run all reinforcement tests"
	@echo "make coverage"
	@echo "       run all reinforcement tests and produce coverage report"


meta:
	python meta.py `git describe --tags --abbrev=0`

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	rm --force .coverage
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

install: meta clean
	pip install --upgrade pip
	pip install --upgrade setuptools
	pip install .[$(TARGET)]

test: install
	pytest --verbose --color=yes .

coverage: install 
	pytest --cov=reinforcement --cov-report term-missing