# Atmosphere

Research system for anchored world model generation via spatial retrieval.

This repo implements the Phase 0 proof-of-concept for Atmosphere.ai — a pipeline that retrieves real-world spatial data (GIS geometry, street-level imagery) and uses it to condition open-source generative world models.

**Status:** Phase 0 Week 1 — data foundation.

## Documents

- `concept_report.md` — what this project is and why (read first)
- `atmosphere_spec.md` — technical architecture (module-level spec)
- `DAY1_SETUP.md` — environment setup instructions for your first day

## Quick start

```bash
# Install uv if you haven't
brew install uv

# Clone and enter
git clone https://github.com/maxyzhu/atmosphere.git
cd atmosphere

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests to verify setup
pytest
```

## Directory structure

```
atmosphere/                # the Python package
├── geo.py                 # coordinate system utilities (WGS84 <-> ENU)
└── (more modules coming)

tests/                     # unit tests mirror the package layout
scripts/                   # CLI entry points (fetch_neighborhood.py etc.)
data/                      # cached GIS + Mapillary (gitignored)
outputs/                   # generated scenes (gitignored)
```

## License

Code: MIT.
Any GIS data fetched via this pipeline inherits its source's license (OSM: ODbL, USGS: public domain, Mapillary: CC-BY-SA).