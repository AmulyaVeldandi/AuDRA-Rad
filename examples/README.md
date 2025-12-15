# AuDRA-Rad Examples

This directory contains example scripts and demonstrations for using AuDRA-Rad with different LLM backends and data sources.

## Directory Structure

### `ollama/` - Local Ollama LLM Examples
Examples using Ollama with local Llama 3.1 8B model for radiology report analysis.

**Files:**
- `test_ollama_simple.py` - Basic Ollama API connection test
- `test_radiology_analysis.py` - Comprehensive radiology report analysis with Fleischner guidelines
- `ollama_radiology_analysis.ipynb` - Interactive notebook for analyzing MIMIC-IV radiology reports

**Prerequisites:**
```bash
# Install and start Ollama
ollama serve

# Pull the model
ollama pull llama3.1:8b

# Install Python dependencies
pip install requests jsonschema tenacity pydantic pydantic-settings loguru
```

**Quick Start:**
```bash
# Test Ollama connection
python examples/ollama/test_ollama_simple.py

# Run full radiology analysis suite
python examples/ollama/test_radiology_analysis.py
```

### `mimic_exploration/` - MIMIC-IV Data Exploration
Examples for exploring and analyzing MIMIC-IV radiology reports.

**Files:**
- `explore_mimic_data.ipynb` - Jupyter notebook for exploring MIMIC-IV radiology dataset

**Data Requirements:**
- MIMIC-IV Clinical Notes (radiology.csv)
- Available from: https://physionet.org/content/mimic-iv-note/

**Analysis Capabilities:**
- CT Chest report filtering
- Pulmonary nodule detection
- Fleischner guideline term identification
- Statistical analysis of report types

## Sample Data

The repository includes sample radiology reports in `data/sample_reports/`:
- `chest_ct_nodule.txt` - CT chest with solid pulmonary nodule
- `chest_ct_ggo.txt` - CT chest with ground-glass opacities
- Corresponding FHIR JSON representations

## Guidelines

Clinical guidelines are available in `data/guidelines/`:
- `fleischner_2017.md` - Fleischner Society 2017 pulmonary nodule recommendations
- `acr_lung.md` - ACR lung nodule management guidelines
- `acr_liver.md` - ACR liver lesion management guidelines

## Related Directories

- `notebooks/` - Original agent demonstration notebooks
  - `01_explore_reports.ipynb` - Report exploration
  - `02_test_retrieval.ipynb` - Guideline retrieval testing
  - `03_agent_demo.ipynb` - Full agent demonstration

- `scripts/` - Deployment and utility scripts
  - `index_guidelines.py` - Index guidelines into vector store
  - `seed_sample_data.py` - Seed sample data
  - `test_nim_connection.py` - Test NVIDIA NIM connection
  - Deployment scripts for EKS

## Running the Examples

### Ollama Examples

1. **Basic Connection Test:**
   ```bash
   python examples/ollama/test_ollama_simple.py
   ```

2. **Full Radiology Analysis:**
   ```bash
   python examples/ollama/test_radiology_analysis.py
   ```

   This runs 5 comprehensive tests:
   - Nodule information extraction
   - Fleischner guideline recommendations
   - Structured JSON extraction
   - Ground-glass opacity analysis
   - Multi-report comparison

3. **Interactive Notebook:**
   ```bash
   jupyter notebook examples/ollama/ollama_radiology_analysis.ipynb
   ```

### MIMIC Exploration

1. **Explore MIMIC-IV Data:**
   ```bash
   jupyter notebook examples/mimic_exploration/explore_mimic_data.ipynb
   ```

## Test Results

All Ollama tests passed successfully with:
- ✓ Accurate nodule extraction
- ✓ Correct Fleischner guideline application
- ✓ Valid JSON structured output
- ✓ Clinical reasoning capabilities
- ✓ Multi-report comparison

## Next Steps

After running the examples:
1. Review the `src/` directory for the main application code
2. Check `tests/` for unit tests
3. See deployment scripts in `scripts/` for production setup
4. Explore the frontend in `frontend/` for the web interface
