# --- Shared theme styles -----------------------------------------------------
# brand-theme.css is loaded LIVE from the deployed site (see _quarto.yml), so it
# never needs copying. The SCSS theme partials, however, are compiled per-site by
# Quarto and must exist locally at build time — they cannot be a remote URL in
# `theme:`. These targets bring them in from the main repo.
#
# SHARED  (fetched/overwritten — do NOT edit): sketchy-light.scss sketchy-dark.scss
# LOCAL   (your overrides — never fetched, safe to edit): custom.scss local-style.css

# Theme partials to keep in sync (only these are ever overwritten).
STYLES := sketchy-light.scss sketchy-dark.scss

# Remote source (raw GitHub) — used by `fetch-styles`.
STYLE_BASE_URL ?= https://raw.githubusercontent.com/migue-rc/migue-rc.github.io/main

# Local source (sibling checkout) — used by `sync-styles`.
MAIN_REPO ?= ../migue-rc.github.io

.PHONY: publish preview fetch-styles sync-styles

# Pull the latest SCSS from GitHub before deploying, then publish.
publish: fetch-styles
	quarto publish gh-pages

preview:
	quarto preview index.ipynb

# Fetch the shared SCSS from the main repo over the network (no local checkout
# needed). Requires the files to be committed/pushed to the main repo first.
fetch-styles:
	@for f in $(STYLES); do \
		echo "fetching $$f ..."; \
		curl -fsSL "$(STYLE_BASE_URL)/$$f" -o "$$f" \
			|| { echo "ERROR: could not fetch $(STYLE_BASE_URL)/$$f" >&2; exit 1; }; \
	done
	@echo "Styles fetched from $(STYLE_BASE_URL)"

# Offline alternative: copy the SCSS from a local checkout of the main repo.
sync-styles:
	@if [ ! -d "$(MAIN_REPO)" ]; then \
		echo "ERROR: main repo not found at '$(MAIN_REPO)'." >&2; \
		echo "       Run: make sync-styles MAIN_REPO=/path/to/migue-rc.github.io" >&2; \
		exit 1; \
	fi
	@for f in $(STYLES); do \
		if [ -f "$(MAIN_REPO)/$$f" ]; then \
			cp "$(MAIN_REPO)/$$f" "./$$f" && echo "synced $$f"; \
		else \
			echo "WARNING: $(MAIN_REPO)/$$f not found, skipped" >&2; \
		fi; \
	done
