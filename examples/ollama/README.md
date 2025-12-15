# Ollama Examples for AuDRA-Rad

This directory contains examples for using Ollama with local Llama 3.1 8B model to analyze radiology reports and apply clinical guidelines.

## Setup

### 1. Install Ollama

Download and install from: https://ollama.ai

### 2. Start Ollama Service

```bash
ollama serve
```

### 3. Pull the Model

```bash
ollama pull llama3.1:8b
```

This will download ~4.9GB of model weights.

### 4. Install Python Dependencies

```bash
pip install requests jsonschema tenacity pydantic pydantic-settings loguru
```

## Files

### `test_ollama_simple.py`

Basic connection test for Ollama API. Verifies:
- Ollama service is running
- Model is available
- Text generation works
- JSON mode works

**Run:**
```bash
python test_ollama_simple.py
```

**Expected output:**
```
✓ Ollama service is running
✓ Text generation successful
✓ JSON generation successful
✓ All Ollama tests passed!
```

### `test_radiology_analysis.py`

Comprehensive test suite for radiology report analysis using sample data from `data/sample_reports/`.

**Features:**
1. **Nodule Information Extraction** - Extract size, location, type, count
2. **Fleischner Guideline Recommendations** - Apply 2017 guidelines for follow-up
3. **Structured JSON Extraction** - Generate machine-readable output
4. **Ground-Glass Opacity Analysis** - Analyze subsolid nodules
5. **Multi-Report Comparison** - Compare different findings

**Run:**
```bash
python test_radiology_analysis.py
```

**Sample Output:**
```json
{
  "has_nodules": true,
  "nodule_count": 1,
  "largest_nodule_mm": 8,
  "nodule_type": "solid",
  "location": "left lower lobe",
  "fleischner_applicable": true,
  "recommended_followup": "CT at 6-12 months",
  "confidence": "high"
}
```

### `ollama_radiology_analysis.ipynb`

Interactive Jupyter notebook for exploring Ollama capabilities with MIMIC-IV radiology data.

**Features:**
- Initialize Ollama client
- Load MIMIC-IV radiology reports
- Filter for CT chest with pulmonary nodules
- Extract nodule information
- Apply Fleischner guidelines
- Batch process multiple reports
- Track token usage and metrics

**Run:**
```bash
jupyter notebook ollama_radiology_analysis.ipynb
```

## Sample Data

The examples use radiology reports from `../../data/sample_reports/`:

**chest_ct_nodule.txt:**
```
Examination: CT Chest with contrast
Findings: Solid pulmonary nodule in the left lower lobe measuring 8 mm.
Impression: Consider Fleischner 2017 guidelines for surveillance.
```

**chest_ct_ggo.txt:**
```
Examination: CT Chest without contrast
Findings: Ground-glass opacities in the right upper lobe measuring 12 mm.
Impression: Suggest follow-up imaging in 3 months to confirm resolution.
```

## Clinical Guidelines

The examples use Fleischner 2017 guidelines from `../../data/guidelines/fleischner_2017.md`.

**Key Recommendations:**

| Nodule Type | Size | Risk | Follow-up |
|-------------|------|------|-----------|
| Solid | < 6mm | Low | No routine follow-up |
| Solid | 6-8mm | Low | CT at 6-12 months |
| Solid | > 8mm | Low | CT at 3 months, consider PET/CT |
| Ground-glass | ≥ 6mm | Any | CT at 6-12 months to confirm persistence |

## Test Results

All tests passed successfully:

✓ **Nodule Extraction**: Correctly identified 8mm solid nodule in left lower lobe
✓ **Fleischner Application**: Recommended CT at 6-12 months for low-risk patient
✓ **JSON Output**: Generated valid structured data
✓ **Ground-Glass Analysis**: Corrected 3-month to 6-12 month follow-up
✓ **Comparison**: Correctly prioritized solid nodule over ground-glass opacity

## Performance Metrics

With Llama 3.1 8B on local hardware:
- Average latency: ~2-5 seconds per request
- Token usage: ~50-200 tokens per analysis
- Accuracy: High confidence on standard cases
- Cost: $0 (runs locally)

## Troubleshooting

### Ollama not connecting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
killall ollama
ollama serve
```

### Model not found
```bash
# List installed models
ollama list

# Pull the model
ollama pull llama3.1:8b
```

### Python import errors
```bash
# Install missing dependencies
pip install requests jsonschema tenacity pydantic pydantic-settings loguru
```

## Integration with AuDRA-Rad

To use Ollama in the main AuDRA-Rad application:

1. Use the `OllamaClient` from `src/services/ollama_llm.py`
2. Configure model and base URL in settings
3. Replace OpenAI/Bedrock calls with Ollama client
4. Handle structured output with JSON mode

**Example:**
```python
from src.services.ollama_llm import OllamaClient

client = OllamaClient(model_name="llama3.1:8b")

# Generate text
response = client.generate(
    prompt="Analyze this radiology report...",
    temperature=0.1,
    max_tokens=1024
)

# Generate structured JSON
data = client.generate_json(
    prompt="Extract nodule information...",
    schema={"type": "object", "properties": {...}}
)
```

## Next Steps

1. Explore the main application code in `src/`
2. Review unit tests in `tests/`
3. Check deployment scripts in `scripts/`
4. Try the web interface in `frontend/`
