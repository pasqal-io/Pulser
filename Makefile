.PHONY: dev-install
dev-install: dev-install-core dev-install-simulation dev-install-pasqal

.PHONY: dev-install-no-torch
dev-install-no-torch: dev-install-core-no-torch dev-install-simulation dev-install-pasqal

.PHONY: dev-install-core
dev-install-core:
	pip install -e ./pulser-core[torch]

.PHONY: dev-install-core-no-torch
dev-install-core-no-torch:
	pip install -e ./pulser-core

.PHONY: dev-install-simulation
dev-install-simulation:
	pip install -e ./pulser-simulation

.PHONY: dev-install-pasqal
dev-install-pasqal:
	pip install -e ./pulser-pasqal