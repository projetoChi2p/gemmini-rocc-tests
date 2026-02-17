# ==========================================
# ViT Model Runner Makefile
# ==========================================
# Usage:
#   make -f RunViT.mk run MODEL=minivit DATASET=sat6
#   make -f RunViT.mk run MODEL=deitvit DATASET=mnist
#   make -f RunViT.mk run MODEL=deitcnn DATASET=cifar10
#
# Available models: minivit, deitvit, deitcnn
# Available datasets: sat6, emnist, mnist, cifar10, cifar100
# ==========================================

# Default values
MODEL ?= minivit
DATASET ?= sat6

# Output directory
OUTPUT_DIR = outputs
OUTPUT_FILE = $(OUTPUT_DIR)/$(MODEL)_$(DATASET)_report.txt

# Build and run the model
.PHONY: run build clean help

run: build
	@echo "========================================"
	@echo "Running $(MODEL) on $(DATASET) dataset"
	@echo "========================================"
	@mkdir -p $(OUTPUT_DIR)
	@echo "Build Configuration:" > $(OUTPUT_FILE)
	@echo "  Model: $(MODEL)" >> $(OUTPUT_FILE)
	@echo "  Dataset: $(DATASET)" >> $(OUTPUT_FILE)
	@echo "  Date: $$(date)" >> $(OUTPUT_FILE)
	@echo "" >> $(OUTPUT_FILE)
	@echo "Running inference..." | tee -a $(OUTPUT_FILE)
	@cd build && spike --extension=gemmini pk ViTRuns/vit_merged-pk >> ../$(OUTPUT_FILE) 2>&1 || \
	 (echo "Note: Using spike simulator. For baremetal, adjust command accordingly." >> ../$(OUTPUT_FILE))
	@echo ""
	@echo "========================================"
	@echo "Results saved to: $(OUTPUT_FILE)"
	@echo "========================================"
	@tail -20 $(OUTPUT_FILE)

run-debug: build
	@echo "========================================"
	@echo "Running $(MODEL) on $(DATASET) dataset (Debug Mode)"
	@echo "========================================"
	@mkdir -p $(OUTPUT_DIR)
	@echo "Build Configuration (Debug):" > $(OUTPUT_FILE)
	@echo "  Model: $(MODEL)" >> $(OUTPUT_FILE)
	@echo "  Dataset: $(DATASET)" >> $(OUTPUT_FILE)
	@echo "  Date: $$(date)" >> $(OUTPUT_FILE)
	@echo "" >> $(OUTPUT_FILE)
	@echo "Running inference in debug mode..." | tee -a $(OUTPUT_FILE)
	@cd build && spike --extension=gemmini pk ViTRuns/vit_merged-pk 
	@echo ""
	@echo "========================================"
	@echo "Results saved to: $(OUTPUT_FILE)"
	@echo "========================================"
	@tail -20 $(OUTPUT_FILE)

build:
	@echo "Building $(MODEL) with $(DATASET) dataset..."
	@MODEL=$(MODEL) DATASET=$(DATASET) ./build.sh

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ViTRuns/vit_merged-*
	@echo "Clean complete. Run 'make build' to rebuild."

clean-all:
	@echo "Cleaning all build artifacts and outputs..."
	@rm -rf build outputs
	@echo "Clean complete."

help:
	@echo "ViT Model Runner"
	@echo ""
	@echo "Usage:"
	@echo "  make -f RunViT.mk run MODEL=<model> DATASET=<dataset>"
	@echo ""
	@echo "Available Models:"
	@echo "  - minivit   : Mini Vision Transformer"
	@echo "  - deitvit   : DeiT Vision Transformer"
	@echo "  - deitcnn   : DeiT with CNN Embedding"
	@echo ""
	@echo "Available Datasets:"
	@echo "  - sat6      : SAT-6 Satellite Dataset"
	@echo "  - emnist    : Extended MNIST"
	@echo "  - mnist     : MNIST Digits"
	@echo "  - cifar10   : CIFAR-10"
	@echo "  - cifar100  : CIFAR-100"
	@echo ""
	@echo "Examples:"
	@echo "  make -f RunViT.mk run MODEL=minivit DATASET=sat6"
	@echo "  make -f RunViT.mk run MODEL=deitvit DATASET=mnist"
	@echo "  make -f RunViT.mk clean"
	@echo ""
	@echo "Output: results saved to outputs/MODEL_DATASET_report.txt"
