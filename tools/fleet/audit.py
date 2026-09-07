"""
AMEVA Fleet Image Audit Utility
===============================
Measures Shannon entropy, dynamic range clipping, and perceptual integrity of generated images.
"""
import math
from typing import Dict, Any

def audit_image(image_path: str) -> Dict[str, Any]:
    """Audit image Shannon entropy and dynamic range clipping."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            gray = im.convert('L')
            hist = gray.histogram()
            total_pixels = gray.width * gray.height

        if total_pixels == 0:
            return {'entropy_bits': 0.0, 'clipped_low_pct': 0.0, 'clipped_high_pct': 0.0}

        clipped_low = (hist[0] / total_pixels) * 100.0
        clipped_high = (hist[255] / total_pixels) * 100.0

        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / total_pixels
                entropy -= p * math.log2(p)

        return {
            'entropy_bits': round(entropy, 2),
            'clipped_low_pct': round(clipped_low, 2),
            'clipped_high_pct': round(clipped_high, 2)
        }
    except Exception as e:
        return {
            'entropy_bits': 0.0,
            'clipped_low_pct': 0.0,
            'clipped_high_pct': 0.0,
            'error': str(e)
        }
