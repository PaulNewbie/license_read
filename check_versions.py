import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import typing
from dataclasses import dataclass
from rapidfuzz import fuzz, process
import tkinter as tk
import difflib

def check_versions():
    """Check versions of all imported libraries"""
    
    libraries = {
        'OpenCV': cv2.__version__,
        'NumPy': np.__version__,
        'Pillow (PIL)': Image.__version__,
        'RapidFuzz': getattr(__import__('rapidfuzz'), '__version__', 'Version not available'),
        'Tkinter': tk.TkVersion,
    }
    
    # Special cases for libraries without direct version attributes
    try:
        import pytesseract
        libraries['PyTesseract'] = pytesseract.__version__
    except AttributeError:
        try:
            import pkg_resources
            libraries['PyTesseract'] = pkg_resources.get_distribution('pytesseract').version
        except:
            libraries['PyTesseract'] = 'Version not directly available'
    
    # Built-in modules (no version info typically)
    import sys
    libraries['Python'] = sys.version.split()[0]
    libraries['re (built-in)'] = 'Built-in module'
    libraries['typing (built-in)'] = 'Built-in module'
    libraries['dataclasses (built-in)'] = 'Built-in module'
    libraries['difflib (built-in)'] = 'Built-in module'
    
    print("Library Versions:")
    print("=" * 40)
    for lib, version in libraries.items():
        print(f"{lib:<20}: {version}")

if __name__ == "__main__":
    check_versions()