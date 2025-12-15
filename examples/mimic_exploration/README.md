# MIMIC-IV Radiology Data Exploration

This directory contains notebooks for exploring and analyzing the MIMIC-IV radiology report dataset.

## About MIMIC-IV

MIMIC-IV (Medical Information Mart for Intensive Care IV) is a freely accessible critical care database containing de-identified health data from patients admitted to the Beth Israel Deaconess Medical Center.

**Website:** https://physionet.org/content/mimic-iv-note/

## Files

### `explore_mimic_data.ipynb`

Interactive Jupyter notebook for exploring MIMIC-IV radiology reports.

**Features:**
- Load radiology reports from CSV
- Statistical analysis of report types
- Filter CT Chest reports
- Identify pulmonary nodule mentions
- Search for Fleischner guideline terms
- Sample report viewing

## Getting Started

### 1. Request Access to MIMIC-IV

1. Visit https://physionet.org/
2. Create an account
3. Complete CITI training course
4. Request access to MIMIC-IV Clinical Notes
5. Sign data use agreement

### 2. Download the Data

After approval, download the radiology notes:
```bash
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimic-iv-note/2.2/note/radiology.csv
```

### 3. Update Notebook Path

Edit the notebook to point to your downloaded CSV file:
```python
df = pd.read_csv('/path/to/mimic-iv-note/note/radiology.csv')
```

### 4. Run the Notebook

```bash
jupyter notebook explore_mimic_data.ipynb
```

## Dataset Statistics

From initial exploration:

- **Total radiology reports**: ~2.3M+ reports
- **CT Chest reports**: 180,449 (7.8%)
- **CT Chest with pulmonary nodules**: 94,776 (52.5% of CT Chest)
- **Reports with Fleischner-related terms**: 12,882 (13.6% of nodule reports)
- **Chest X-ray (PA and LAT)**: 70,082

## Analysis Examples

### 1. Count Report Types

```python
# Count different examination types
exam_types = df['text'].str.extract(r'EXAMINATION:\s+([^\n]+)')
exam_types[0].value_counts().head(10)
```

### 2. Filter for Specific Findings

```python
# Find CT Chest reports with pulmonary nodules
ct_chest = df[df['text'].str.contains('CT CHEST|CT OF THE CHEST|CHEST CT',
                                       case=False, na=False)]
nodule_terms = 'nodule|nodular|pulmonary nodule'
ct_chest_nodules = ct_chest[ct_chest['text'].str.contains(nodule_terms,
                                                           case=False, na=False)]
```

### 3. Search for Guidelines

```python
# Find reports mentioning Fleischner guidelines
fleischner_keywords = 'fleischner|follow-up recommended|mm nodule'
fleischner_reports = df[df['text'].str.contains(fleischner_keywords,
                                                 case=False, na=False)]
```

## Sample Report

Example CT Chest report with pulmonary nodule and Fleischner recommendation:

```
HISTORY: Patient with fevers, nausea, vomiting, abdominal pain

FINDINGS:
LUNG BASES: Perifissural nodule is seen on the uppermost slice on the right
major fissure measuring 3 mm. A right lower lobe nodule measures 3mm. There
is bibasilar atelectasis, but no pleural effusion.

IMPRESSION:
1. No acute intra-abdominal pathology.
2. 3-mm nodule seen along the right major fissure and right lower lobe.
According to Fleischner guidelines, in the absence of risk factors, no further
followup is needed. If patient has risk factors such as smoking, followup
chest CT at 12 months is recommended to document stability.
```

## Common Findings

### Nodule Size Distribution
- < 6mm: Most common (typically no follow-up needed)
- 6-8mm: Moderate (requires surveillance)
- > 8mm: Less common (requires close follow-up or intervention)

### Follow-up Recommendations
- "No routine follow-up": ~40%
- "CT at 12 months": ~30%
- "CT at 6 months": ~20%
- "CT at 3 months": ~10%

## Use Cases

### 1. Training Data Collection
Filter reports with specific findings for machine learning:
```python
# Get reports with specific characteristics for training
training_data = df[
    (df['text'].str.contains('solid nodule', case=False)) &
    (df['text'].str.contains('follow-up', case=False))
]
```

### 2. Guideline Adherence Analysis
Check if recommendations follow Fleischner guidelines:
```python
# Extract nodule size and recommendations
reports = df[df['text'].str.contains('mm nodule', case=False)]
# Parse and validate against guidelines
```

### 3. Benchmark Dataset Creation
Create test cases for AuDRA-Rad:
```python
# Sample diverse cases for testing
test_cases = ct_chest_nodules.sample(100)
# Export for evaluation
test_cases.to_csv('test_cases.csv')
```

## Privacy and Ethics

**Important:** MIMIC-IV data is de-identified but still contains sensitive medical information.

- ✓ Only use for research purposes
- ✓ Never attempt to re-identify patients
- ✓ Follow data use agreement terms
- ✓ Cite MIMIC-IV in publications
- ✗ Do not share raw data publicly
- ✗ Do not use for commercial purposes without permission

## Citation

If you use MIMIC-IV in your research:

```
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67.

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ...
& Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a
new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.
```

## Related Resources

- **MIMIC-IV Documentation**: https://mimic.mit.edu/docs/iv/
- **MIMIC Code Repository**: https://github.com/MIT-LCP/mimic-code
- **PhysioNet**: https://physionet.org/
- **MIMIC-IV Papers**: https://mimic.mit.edu/docs/about/publications/

## Integration with AuDRA-Rad

The exploration insights can be used to:

1. **Identify common patterns** in radiology reports
2. **Create test datasets** with known ground truth
3. **Validate guideline retrieval** against real-world cases
4. **Benchmark agent performance** on diverse findings
5. **Train custom models** for entity extraction

## Next Steps

1. Filter reports by specific modality (CT, MRI, X-ray)
2. Extract structured findings programmatically
3. Create annotated dataset for evaluation
4. Develop automated guideline compliance checker
5. Build benchmark test suite from MIMIC reports
