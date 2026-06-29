"""Parameter extractor for advanced NLP-based scenario analysis."""

from typing import Dict, List
import re


class ParameterExtractor:
    """Extract numerical parameters from text using regex and NLP."""
    
    def __init__(self):
        """Initialize the parameter extractor."""
        pass
    
    def extract_percentage(self, text: str) -> float:
        """Extract percentage values from text."""
        pattern = r'(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[0]) / 100.0
        return 0.0
    
    def extract_duration(self, text: str) -> int:
        """Extract duration in days from text."""
        patterns = [
            r'(\d+)\s*(?:days?|d)',
            r'(\d+)\s*(?:weeks?|w)\s*->\s*\1 * 7',
            r'(\d+)\s*(?:months?|m)\s*->\s*\1 * 30'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return int(matches[0])
        return 30  # default
