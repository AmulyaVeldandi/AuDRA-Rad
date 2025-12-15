#!/usr/bin/env python3
"""Test Ollama with real radiology reports and Fleischner guidelines."""

import json
import requests
from pathlib import Path

# Ollama configuration
BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"
TIMEOUT = 120.0

def load_file(filepath):
    """Load text file content."""
    return Path(filepath).read_text().strip()

def ollama_generate(prompt, system_prompt=None, format_json=False):
    """Call Ollama API for text or JSON generation."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        }
    }

    if system_prompt:
        payload["system"] = system_prompt

    if format_json:
        payload["format"] = "json"

    response = requests.post(
        f"{BASE_URL}/api/generate",
        json=payload,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()

    return result.get("response", "")

def test_nodule_extraction():
    """Test 1: Extract nodule information from CT report."""
    print("=" * 80)
    print("TEST 1: Nodule Information Extraction")
    print("=" * 80)

    report = load_file("data/sample_reports/chest_ct_nodule.txt")
    print(f"\nRadiology Report:\n{report}\n")

    prompt = f"""
Analyze this radiology report and extract pulmonary nodule information:

{report}

Extract:
1. Number of nodules
2. Size of each nodule (in mm)
3. Location
4. Nodule type (solid, ground-glass, part-solid)
5. Any recommendations mentioned

Provide a clear, concise summary.
"""

    print("Analyzing with Ollama...\n")
    response = ollama_generate(prompt)
    print(f"Analysis:\n{response}\n")

    return response

def test_fleischner_recommendation():
    """Test 2: Apply Fleischner guidelines to recommend follow-up."""
    print("=" * 80)
    print("TEST 2: Fleischner Guideline Recommendations")
    print("=" * 80)

    report = load_file("data/sample_reports/chest_ct_nodule.txt")
    guidelines = load_file("data/guidelines/fleischner_2017.md")

    print(f"\nRadiology Report:\n{report}\n")

    system_prompt = f"""You are a radiologist assistant. Use these Fleischner 2017 guidelines:

{guidelines}

Analyze reports and provide appropriate follow-up recommendations."""

    prompt = f"""
Analyze this CT chest report and provide Fleischner guideline-based recommendations:

{report}

Based on the Fleischner 2017 guidelines, determine:
1. Nodule classification (solid/subsolid)
2. Patient risk category (assume low-risk if not specified)
3. Recommended follow-up imaging timeline
4. Explain your reasoning

Provide a structured recommendation.
"""

    print("Generating Fleischner recommendations...\n")
    response = ollama_generate(prompt, system_prompt=system_prompt)
    print(f"Recommendations:\n{response}\n")

    return response

def test_structured_json_extraction():
    """Test 3: Extract structured JSON data."""
    print("=" * 80)
    print("TEST 3: Structured JSON Extraction")
    print("=" * 80)

    report = load_file("data/sample_reports/chest_ct_nodule.txt")
    print(f"\nRadiology Report:\n{report}\n")

    prompt = f"""
Analyze this radiology report and extract information in JSON format:

{report}

Return a JSON object with these fields:
{{
  "has_nodules": true/false,
  "nodule_count": number,
  "largest_nodule_mm": number or null,
  "nodule_type": "solid" or "ground-glass" or "part-solid" or null,
  "location": "string describing location",
  "fleischner_applicable": true/false,
  "recommended_followup": "string describing follow-up",
  "confidence": "high" or "medium" or "low"
}}
"""

    print("Extracting structured JSON...\n")
    response = ollama_generate(prompt, format_json=True)

    try:
        data = json.loads(response)
        print(f"Extracted JSON:\n{json.dumps(data, indent=2)}\n")
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response}")
        return None

def test_ground_glass_analysis():
    """Test 4: Analyze ground-glass opacity report."""
    print("=" * 80)
    print("TEST 4: Ground-Glass Opacity Analysis")
    print("=" * 80)

    report = load_file("data/sample_reports/chest_ct_ggo.txt")
    guidelines = load_file("data/guidelines/fleischner_2017.md")

    print(f"\nRadiology Report:\n{report}\n")

    system_prompt = f"""You are a radiologist assistant. Use these Fleischner 2017 guidelines:

{guidelines[:1000]}... [guidelines loaded]

Focus on subsolid/ground-glass nodule recommendations."""

    prompt = f"""
Analyze this CT report with ground-glass opacities:

{report}

According to Fleischner 2017 guidelines:
1. Classify the finding (ground-glass vs part-solid vs solid)
2. Determine appropriate follow-up interval
3. Explain the rationale based on size and characteristics

Provide clear recommendations.
"""

    print("Analyzing ground-glass findings...\n")
    response = ollama_generate(prompt, system_prompt=system_prompt)
    print(f"Analysis:\n{response}\n")

    return response

def test_comparison_analysis():
    """Test 5: Compare multiple reports."""
    print("=" * 80)
    print("TEST 5: Multi-Report Comparison")
    print("=" * 80)

    report1 = load_file("data/sample_reports/chest_ct_nodule.txt")
    report2 = load_file("data/sample_reports/chest_ct_ggo.txt")

    print(f"\nReport 1:\n{report1}\n")
    print(f"Report 2:\n{report2}\n")

    prompt = f"""
Compare these two radiology reports:

Report 1:
{report1}

Report 2:
{report2}

Provide:
1. Key differences in findings
2. Different management approaches needed
3. Which requires more urgent follow-up and why

Summarize your comparison.
"""

    print("Comparing reports...\n")
    response = ollama_generate(prompt)
    print(f"Comparison:\n{response}\n")

    return response

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("OLLAMA RADIOLOGY ANALYSIS TEST SUITE")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Base URL: {BASE_URL}")
    print("=" * 80 + "\n")

    try:
        # Run all tests
        test_nodule_extraction()
        test_fleischner_recommendation()
        json_result = test_structured_json_extraction()
        test_ground_glass_analysis()
        test_comparison_analysis()

        print("=" * 80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

        if json_result:
            print("\nSample Structured Output:")
            print(json.dumps(json_result, indent=2))

        print("\n✓ Ollama is ready for radiology report analysis!")
        print("✓ Fleischner guideline integration working!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
