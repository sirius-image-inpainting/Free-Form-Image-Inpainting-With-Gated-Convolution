SHELL:=bash
.PHONY: train


TRAIN_DATASET_URL=http://data.csail.mit.edu/places/places365/train_256_places365standard.tar
VALID_DATASET_URL=http://data.csail.mit.edu/places/places365/val_256.tar
TEST_DATASET_URL=http://data.csail.mit.edu/places/places365/test_256.tar


download_train:
	mkdir -p ./data/compressed/
	axel -n 60 $(TRAIN_DATASET_URL) --output=./data/compressed/train.tar

download_valid:
	mkdir -p ./data/compressed/
	axel -n 60 $(VALID_DATASET_URL) --output=./data/compressed/valid.tar

download_test:
	mkdir -p ./data/compressed/
	axel -n 60 $(TEST_DATASET_URL) --output=./data/compressed/test.tar

install_venv:
	virtualenv --python=python3 venv
	source venv/bin/activate
	pip install -r requirements.txt

clean:
	rm -rf venv/

train:
	PYTHONPATH=. python3 cmd/train.py --config config.yaml

summary:
	PYTHONPATH=. python3 cmd/model_summary.py
