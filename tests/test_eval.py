# tests/test_eval.py
import pytest
from src.evaluate import calculate_accuracy

def test_model_accuracy():
    accuracy = calculate_accuracy()
    assert accuracy > 0.90, f"Model accuracy {accuracy} is below threshold"
