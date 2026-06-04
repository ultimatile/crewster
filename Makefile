# Makefile for crewster release management

SHELL := /bin/bash
.PHONY: help release-full release publish restore-branch clean-env
.PHONY: version-bump git-tag git-push release-summary
.PHONY: test typecheck lint check clean-artifacts
.DEFAULT_GOAL := help

# Terminal colors
RED := $(shell tput setaf 1)
GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
CYAN := $(shell tput setaf 6)
GRAY := $(shell tput setaf 8)
RESET := $(shell tput sgr0)

# Get version information
CURRENT_VERSION := $(shell gsed -n -E 's/.*(version[[:space:]]*=[[:space:]]*")([^"]*)(")/\2/p' pyproject.toml)
NEXT_VERSION := $(shell uv run semantic-release version --print 2>/dev/null || echo "")
ORIGINAL_BRANCH := $(shell git branch --show-current)

# Validate version change
ifeq ($(NEXT_VERSION),)
VERSION_VALID := false
else ifeq ($(NEXT_VERSION),$(CURRENT_VERSION))
VERSION_VALID := false
else
VERSION_VALID := true
RELEASE_VERSION := $(NEXT_VERSION)
endif

help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Release workflow targets
release-full: release publish restore-branch ## Complete release workflow (prepare + publish + restore)
	@echo "$(GREEN)Full release workflow completed$(RESET)"

release: version-bump git-tag git-push release-summary ## Complete release preparation workflow

publish: ## Create GitHub release
	$(eval LATEST_TAG := $(shell git describe --tags --abbrev=0 2>/dev/null || echo ""))
	$(eval PUB_VERSION := $(if $(RELEASE_VERSION),$(RELEASE_VERSION),$(patsubst v%,%,$(LATEST_TAG))))
	@if [ -z "$(PUB_VERSION)" ]; then \
		echo "$(RED)No version found. Run 'make release' first or ensure tags exist.$(RESET)"; \
		exit 1; \
	fi
	@if ! ./scripts/confirm.sh "Create GitHub release for v$(PUB_VERSION)?"; then \
		echo "$(RED)GitHub release creation cancelled$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)GitHub release creation confirmed for v$(PUB_VERSION)$(RESET)"
	@if command -v gh &> /dev/null; then \
		echo "$(CYAN)Creating GitHub release...$(RESET)"; \
		gh release create "v$(PUB_VERSION)" --generate-notes; \
		echo "$(GREEN)GitHub release v$(PUB_VERSION) published$(RESET)"; \
	else \
		echo "$(YELLOW)GitHub CLI not found. Create release manually:$(RESET)"; \
		echo "   https://github.com/ultimatile/hpc/releases/new?tag=v$(PUB_VERSION)"; \
	fi

version-bump: ## Update version in pyproject.toml
	@if [ "$(VERSION_VALID)" != "true" ]; then \
		echo "$(RED)No version change detected$(RESET)"; \
		echo "$(YELLOW)Make sure you have semantic commits (feat:, fix:, etc.)$(RESET)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Version bump will be v$(CURRENT_VERSION) → v$(NEXT_VERSION)$(RESET)"
	@if ! ./scripts/confirm.sh "Proceed with release v$(NEXT_VERSION)?"; then \
		echo "$(RED)Release cancelled$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Release v$(NEXT_VERSION) confirmed$(RESET)"
	@if [ "$(ORIGINAL_BRANCH)" != "main" ]; then \
		echo "$(CYAN)Switching to main branch...$(RESET)"; \
		git switch main; \
		echo "$(GREEN)Switched to main branch$(RESET)"; \
	fi
	@$(MAKE) check
	@echo "$(CYAN)Updating version to $(NEXT_VERSION)...$(RESET)"
	@gsed -i -E 's/(version[[:space:]]*=[[:space:]]*")[^"]*(")/\1$(NEXT_VERSION)\2/' pyproject.toml
	@uv sync
	@echo "$(GREEN)Version updated to $(NEXT_VERSION)$(RESET)"

git-tag: ## Create release commit and tag
	@echo "$(CYAN)Creating release commit and tag...$(RESET)"
	@git add pyproject.toml uv.lock
	@git commit -m "release: v$(RELEASE_VERSION)"
	@git tag "v$(RELEASE_VERSION)"
	@echo "$(GREEN)Release v$(RELEASE_VERSION) prepared locally$(RESET)"

git-push: ## Push commits and tags to GitHub
	@echo "$(CYAN)Pushing to GitHub...$(RESET)"
	@git push origin main
	@git push origin v$(RELEASE_VERSION)
	@echo "$(GREEN)Pushed v$(RELEASE_VERSION) to GitHub$(RESET)"

release-summary: ## Show release summary and next steps
	@echo
	@echo "$(GREEN)Release v$(RELEASE_VERSION) prepared successfully$(RESET)"
	@echo
	@echo "Next steps:"
	@echo "1. Create GitHub release: make publish"
	@echo "2. Restore original branch: make restore-branch"

restore-branch: ## Restore original branch
	@if [ "$(ORIGINAL_BRANCH)" != "main" ]; then \
		echo "$(CYAN)Restoring original branch: $(ORIGINAL_BRANCH)$(RESET)"; \
		git checkout "$(ORIGINAL_BRANCH)"; \
		echo "$(GREEN)Restored to $(ORIGINAL_BRANCH)$(RESET)"; \
	else \
		echo "$(GRAY)Staying on main branch$(RESET)"; \
	fi

clean-env: ## Clean environment (placeholder for compatibility)
	@echo "$(GREEN)Environment cleaned$(RESET)"

# Development targets
test: ## Run all tests
	@uv run pytest -v

typecheck: ## Run type checking
	@uv run pyright src/crewster/

lint: ## Run linting and formatting
	@uv run ruff check
	@uv run ruff format

check: test typecheck lint ## Run all checks (test + typecheck + lint)

clean-artifacts: ## Clean build artifacts
	@rm -rf dist/
	@rm -rf test_output/
	@find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name '*.pyc' -delete 2>/dev/null || true
	@echo "$(GREEN)Cleaned build artifacts$(RESET)"

# Aliases for convenience
t: test ## Alias for test
c: typecheck ## Alias for typecheck  
l: lint ## Alias for lint
tcl: check ## Alias for check

# Debug targets
debug-vars: ## Show current version variables
	@echo "CURRENT_VERSION: $(CURRENT_VERSION)"
	@echo "NEXT_VERSION: $(NEXT_VERSION)"
	@echo "RELEASE_VERSION: $(RELEASE_VERSION)"
	@echo "VERSION_VALID: $(VERSION_VALID)"
	@echo "ORIGINAL_BRANCH: $(ORIGINAL_BRANCH)"
