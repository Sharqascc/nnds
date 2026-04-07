PYTHON ?= python

install:
	pip install -r requirements.txt

smoke-readme:
	$(PYTHON) -c "from pathlib import Path; assert Path('README.md').exists(); assert Path('CONTRIBUTING.md').exists(); print('docs ok')"

smoke-imports:
	$(PYTHON) -c "import pandas, numpy, torch; print('core imports ok')"

check:
	$(MAKE) smoke-readme
	$(MAKE) smoke-imports
