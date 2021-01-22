PWD=$$(pwd)

port=9999
.PHONY: notebook
notebook:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) jupyter-notebook --ip 0.0.0.0 --port $(port) --allow-root

.PHONY: train
train:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) python src/train.py

.PHONY: test
test:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) python src/test.py

.PHONY: python
python:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) python $(fname)

.PHONY: setup
setup:
	pip install -r requirements.txt
