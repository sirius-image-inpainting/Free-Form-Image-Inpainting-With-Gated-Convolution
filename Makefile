SHELL:=bash
.PHONY: train


train:
	python3 cmd/train.py --config config.yaml
