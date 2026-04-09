"""
Step 10: End-to-end verification of the rasterizer on all 7 scenes.

Prints the final verification table comparing:
- mmIR ray tracer (gold reference from Step 0)
- Rasterizer (this module)
- GT (ground truth radar measurements)
"""

import os
import sys

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm25DGS_v2.rasterizer import run_all_scenes

if __name__ == '__main__':
    results = run_all_scenes()
