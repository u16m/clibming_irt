PYTHON ?= uv run python
UV ?= uv
MISE ?= mise

DATA ?= data/ifsc_full
OUTPUT_TRAIN ?= results/patternB
OUTPUT_PREDICT ?= results/predict
MISSING ?= nan
ROUND ?= final
EPOCHS ?= 4000
LR ?= 0.03
MIN_ATTEMPTS ?= 5

.PHONY: setup train predict

setup:
	$(MISE) install
	$(UV) sync

train:
	$(PYTHON) main.py \
		--data $(DATA) \
		--missing $(MISSING) \
		--round $(ROUND) \
		--output $(OUTPUT_TRAIN) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--min_attempts $(MIN_ATTEMPTS)

predict:
	$(PYTHON) main.py \
		--data $(DATA) \
		--missing $(MISSING) \
		--round all \
		--output $(OUTPUT_PREDICT) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--min_attempts $(MIN_ATTEMPTS)
