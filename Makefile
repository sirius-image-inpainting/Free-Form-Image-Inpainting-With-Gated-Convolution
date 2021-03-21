SHELL:=bash
.PHONY: train


TRAIN_DATASET_URL=http://data.csail.mit.edu/places/places365/train_256_places365standard.tar
VALID_DATASET_URL=http://data.csail.mit.edu/places/places365/val_256.tar
TEST_DATASET_URL=http://data.csail.mit.edu/places/places365/test_256.tar


download_train:
	mkdir -p ./data/compressed/
	axel -n 60 $(TRAIN_DATASET_URL) --output=./data/compressed/train.tar
	tar -xf ./data/compressed/train.tar -C ./data
	mv ./data/data_256/ ./data/train/

download_valid:
	mkdir -p ./data/compressed/
	axel -n 60 $(VALID_DATASET_URL) --output=./data/compressed/valid.tar
	tar -xf ./data/compressed/valid.tar -C ./data
	mv ./data/val_256/ ./data/valid/

download_test:
	mkdir -p ./data/compressed/
	axel -n 60 $(TEST_DATASET_URL) --output=./data/compressed/test.tar
	tar -xf ./data/compressed/test.tar -C ./data
	mv ./data/test_256/ ./data/test/

install_venv:
	virtualenv --python=python3 venv
	source venv/bin/activate
	pip install -r requirements.txt

clean:
	rm -rf venv/

train:
	PYTHONPATH=. python3 cmd/train.py --config config.yaml

