# =============================================================================
# AI-Driven Dynamic Hotel Pricing System
#
#   make help        list every target
#   make up          bring the whole stack up in Docker
#   make bootstrap   set a local (non-Docker) environment up from nothing
#
# Every target is also a plain command you can run yourself -- nothing here
# hides anything, it just saves typing.
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON  ?= python
COMPOSE ?= docker compose

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Local development
# --------------------------------------------------------------------------- #

.PHONY: install
install:  ## Install runtime + dev dependencies (editable)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: install-all
install-all:  ## Install everything, including the optional AI agent
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,agent]"

.PHONY: config
config:  ## Validate .env and print the resolved configuration
	$(PYTHON) scripts/check_config.py

.PHONY: db
db:  ## Create the database schema
	$(PYTHON) -m database.init_db

.PHONY: generate
generate:  ## Generate the synthetic dataset into data/synthetic
	$(PYTHON) scripts/generate_data.py

.PHONY: seed
seed:  ## Load the synthetic dataset into PostgreSQL
	$(PYTHON) scripts/seed_database.py

.PHONY: topics
topics:  ## Create the Kafka topics
	$(PYTHON) scripts/create_topics.py

.PHONY: features
features:  ## Build the feature matrix and store it
	$(PYTHON) scripts/build_features.py

.PHONY: train
train:  ## Train both models and write versioned artifacts
	$(PYTHON) scripts/train_models.py

.PHONY: monitor
monitor:  ## Run data quality and model health checks
	$(PYTHON) scripts/monitor.py

.PHONY: bootstrap
bootstrap:  ## Full local setup, skipping whatever is already done
	$(PYTHON) scripts/bootstrap.py
	@echo ""
	@echo "Ready. Start the API with 'make api' and the dashboard with 'make dashboard'."

.PHONY: bootstrap-check
bootstrap-check:  ## Report which setup steps are done, change nothing
	$(PYTHON) scripts/bootstrap.py --check

.PHONY: reset
reset:  ## Rebuild data, features and models from scratch. Discards pricing history.
	$(PYTHON) scripts/bootstrap.py --force

# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

.PHONY: api
api:  ## Run the API with autoreload
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: dashboard
dashboard:  ## Run the Streamlit dashboard
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

.PHONY: demo-ota
demo-ota:  ## Serve the demo OTA that the competitor scraper scrapes
	$(PYTHON) -m uvicorn demo_ota.app:app --host 0.0.0.0 --port 8900

.PHONY: producer
producer:  ## Collect competitor prices into Kafka (scrapes demo-ota by default)
	$(PYTHON) scripts/run_producer.py

.PHONY: consumer
consumer:  ## Consume Kafka events into PostgreSQL
	$(PYTHON) scripts/run_consumer.py

.PHONY: triage
triage:  ## Summarise the latest monitoring report. Needs the [agent] extra.
	$(PYTHON) -m ai_agent.triage

# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #

.PHONY: test
test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-cov
test-cov:  ## Run the tests with a coverage report
	$(PYTHON) -m pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

.PHONY: lint
lint:  ## Check for unused imports and undefined names
	$(PYTHON) -m pyflakes ai_agent api config dashboard database demo_ota domain features \
	    ingestion models monitoring pricing scripts streaming training tests

.PHONY: check
check: lint test  ## Lint and test

# --------------------------------------------------------------------------- #
# Docker
# --------------------------------------------------------------------------- #

.PHONY: up
up:  ## Build and start the whole stack
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  API        http://localhost:8000/docs"
	@echo "  Dashboard  http://localhost:8501"
	@echo ""
	@echo "  'make logs' to follow, 'make down' to stop."

.PHONY: down
down:  ## Stop the stack, keeping the data
	$(COMPOSE) down

.PHONY: clean
clean:  ## Stop the stack and delete its volumes. Destroys all data.
	$(COMPOSE) down -v

.PHONY: logs
logs:  ## Follow the logs of every service
	$(COMPOSE) logs -f

.PHONY: ps
ps:  ## Show service status
	$(COMPOSE) ps

.PHONY: rebuild
rebuild:  ## Rebuild the application image without cache
	$(COMPOSE) build --no-cache

.PHONY: kafka-image
kafka-image:  ## Build Kafka locally when the registry is unreachable
	docker build -t apache/kafka:4.1.2 docker/kafka

.PHONY: shell
shell:  ## Open a shell inside the API container
	$(COMPOSE) exec api bash
