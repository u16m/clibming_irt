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
RAW_DIR ?= data/ifsc_raw
SCRAPE_OUTPUT ?= data/ifsc_full_scrape.json
SCRAPE_SLEEP ?= 1.5
SCRAPE_RETRIES ?= 5

.PHONY: setup train predict scrape

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


scrape:
	$(PYTHON) -m ifsc_irt_experiment.scrape_cli \
		--raw-dir $(RAW_DIR) \
		--scrape-output $(SCRAPE_OUTPUT) \
		--scrape-sleep $(SCRAPE_SLEEP) \
		--scrape-retries $(SCRAPE_RETRIES)
