SHELL := /bin/bash

setup-python-env: setup-venv activate-venv install-deps

test: lint local-install
	pytest -vv

unit-tests: lint local-install
	pytest -vv ./tests/unit

integration-tests: lint local-install
	pytest -vv ./tests/integration

e2e-tests: lint local-install
	pytest -vv ./tests/e2e

lint:
	flake8 --exclude=.tox,*.egg,venv,.venv,build,data --ignore=W291,E303,E501 --select=E,W,F  .

code-coverage: lint local-install
	pytest -vv --cov --cov-report=term-missing

pkg-clean: 
	rm -Rf ./build; rm -Rf ./dist; rm -Rf ./.tox; rm -Rf ./.eggs; rm -Rf ./.pytest_cache; rm -Rf ./**/*.egg-info; rm -Rf ./**/version.py; rm -Rf ./**/__pycache__; rm -Rf ./**/**/__pycache__; rm -Rf ./htmlcov; rm -Rf ./.coverage; rm -Rf ./*.xml; 

local-install: pkg-clean
	pip3 install -e .

clean-venv:
	rm -Rf .venv

setup-venv:
	python3 -m venv .venv --without-pip

activate-venv:
	pwd && source .venv/bin/activate

install-deps:
	pip3 install --upgrade pip && pip3 install -r requirements.txt