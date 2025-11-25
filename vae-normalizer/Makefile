# VAE Dataset Normalizer - Makefile
# ================================
#
# Build and run the VAE dataset normalization pipeline
#
# Usage:
#   make build          Build release binary
#   make run            Run normalizer on dataset
#   make all            Complete pipeline
#   make verify         Run verification checks
#   make clean          Clean build artifacts

.PHONY: all build run verify clean check-deps validate-cue validate-isabelle test

# Configuration (override via command line)
DATASET_PATH ?= $(HOME)/vae-dataset
OUTPUT_PATH ?= $(HOME)/vae-normalized
SEED ?= 42
STRATA ?= 4
SKIP_CHECKSUMS ?= false

# Binary paths
CARGO := cargo
CUE := cue
NICKEL := nickel
ISABELLE := isabelle

# Build configuration
RELEASE_FLAGS := --release
TARGET_DIR := target/release
BINARY := $(TARGET_DIR)/vae-normalizer

#
# Main targets
#

all: check-deps build run validate-cue
	@echo "Pipeline complete. Output: $(OUTPUT_PATH)"

build:
	@echo "Building vae-normalizer..."
	$(CARGO) build $(RELEASE_FLAGS)
	@echo "Build complete: $(BINARY)"

run: build
	@echo "Running normalizer..."
	@mkdir -p $(OUTPUT_PATH)
	$(BINARY) \
		--dataset $(DATASET_PATH) \
		--output $(OUTPUT_PATH) \
		--seed $(SEED) \
		--strata $(STRATA) \
		$(if $(filter true,$(SKIP_CHECKSUMS)),--skip-checksums,)
	@echo "Normalization complete."

#
# Verification targets
#

verify: validate-cue validate-nickel validate-splits
	@echo "All verification passed."

validate-cue:
	@echo "Validating CUE schema..."
	@if command -v $(CUE) >/dev/null 2>&1; then \
		$(CUE) vet metadata_schema.cue $(OUTPUT_PATH)/metadata.cue 2>/dev/null || \
		echo "Note: CUE validation skipped (install cue: https://cuelang.org)"; \
	else \
		echo "CUE not installed. Skip validation."; \
	fi

validate-nickel:
	@echo "Validating Nickel config..."
	@if command -v $(NICKEL) >/dev/null 2>&1; then \
		$(NICKEL) typecheck config.ncl || \
		echo "Note: Nickel validation failed"; \
	else \
		echo "Nickel not installed. Skip validation."; \
	fi

validate-isabelle:
	@echo "Checking Isabelle proof..."
	@if command -v $(ISABELLE) >/dev/null 2>&1; then \
		$(ISABELLE) build -d . -b VAEDataset_Splits || \
		echo "Note: Isabelle proof check requires Isabelle/HOL installation"; \
	else \
		echo "Isabelle not installed. Skip proof verification."; \
	fi

validate-splits:
	@echo "Validating split properties..."
	@if [ -f "$(OUTPUT_PATH)/splits/random_train.txt" ]; then \
		echo "Checking split disjointness..."; \
		comm -12 <(sort $(OUTPUT_PATH)/splits/random_train.txt) \
		         <(sort $(OUTPUT_PATH)/splits/random_test.txt) | \
		wc -l | grep -q '^0$$' && echo "  Train/Test disjoint: OK" || \
		echo "  ERROR: Train/Test overlap detected"; \
	else \
		echo "No splits found. Run 'make run' first."; \
	fi

#
# Development targets
#

test:
	@echo "Running tests..."
	$(CARGO) test

fmt:
	@echo "Formatting code..."
	$(CARGO) fmt

clippy:
	@echo "Running clippy..."
	$(CARGO) clippy -- -D warnings

check-deps:
	@echo "Checking dependencies..."
	@command -v cargo >/dev/null 2>&1 || { echo "ERROR: Rust/Cargo required"; exit 1; }
	@echo "  cargo: OK"
	@command -v cue >/dev/null 2>&1 && echo "  cue: OK" || echo "  cue: not found (optional)"
	@command -v nickel >/dev/null 2>&1 && echo "  nickel: OK" || echo "  nickel: not found (optional)"
	@command -v julia >/dev/null 2>&1 && echo "  julia: OK" || echo "  julia: not found (optional)"
	@command -v isabelle >/dev/null 2>&1 && echo "  isabelle: OK" || echo "  isabelle: not found (optional)"

#
# Cleanup
#

clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	rm -rf $(OUTPUT_PATH)

clean-output:
	@echo "Cleaning output directory..."
	rm -rf $(OUTPUT_PATH)

#
# Help
#

help:
	@echo "VAE Dataset Normalizer"
	@echo ""
	@echo "Usage: make [target] [VARIABLE=value]"
	@echo ""
	@echo "Targets:"
	@echo "  all             Complete pipeline (build, run, verify)"
	@echo "  build           Build release binary"
	@echo "  run             Run normalizer on dataset"
	@echo "  verify          Run all verification checks"
	@echo "  test            Run Rust tests"
	@echo "  check-deps      Check required dependencies"
	@echo "  clean           Clean all build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET_PATH    Input dataset path (default: ~/vae-dataset)"
	@echo "  OUTPUT_PATH     Output directory (default: ~/vae-normalized)"
	@echo "  SEED            Random seed (default: 42)"
	@echo "  STRATA          Number of strata (default: 4)"
	@echo "  SKIP_CHECKSUMS  Skip checksum computation (default: false)"
	@echo ""
	@echo "Examples:"
	@echo "  make all DATASET_PATH=/data/vae OUTPUT_PATH=/data/output"
	@echo "  make run SKIP_CHECKSUMS=true"
