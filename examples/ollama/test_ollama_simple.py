#!/usr/bin/env python3
"""Simple test script for Ollama without importing config."""

import json
import requests

def test_ollama():
    """Test Ollama API directly."""
    base_url = "http://localhost:11434"
    model_name = "llama3.1:8b"

    print("Testing Ollama connection...")

    # Test 1: Check if Ollama is running
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        print(f"✓ Ollama service is running")
        print(f"  Available models: {[m['name'] for m in models]}")
    except Exception as e:
        print(f"✗ Ollama service check failed: {e}")
        return False

    # Test 2: Simple text generation
    try:
        print(f"\nTesting text generation with {model_name}...")
        payload = {
            "model": model_name,
            "prompt": "Say hello in one sentence.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50,
            }
        }

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()

        print(f"✓ Text generation successful")
        print(f"  Response: {result.get('response', '')[:150]}")
        print(f"  Prompt tokens: {result.get('prompt_eval_count', 0)}")
        print(f"  Completion tokens: {result.get('eval_count', 0)}")

    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        return False

    # Test 3: JSON generation
    try:
        print(f"\nTesting JSON generation...")
        payload = {
            "model": model_name,
            "prompt": "Return a JSON object with fields: name (string), age (number), city (string). Make up realistic data.",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
            }
        }

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()

        json_data = json.loads(result.get('response', '{}'))
        print(f"✓ JSON generation successful")
        print(f"  JSON: {json.dumps(json_data, indent=2)}")

    except Exception as e:
        print(f"✗ JSON generation failed: {e}")
        return False

    print(f"\n✓ All Ollama tests passed! Ready for radiology analysis.")
    return True

if __name__ == "__main__":
    import sys
    success = test_ollama()
    sys.exit(0 if success else 1)
