"""Run this on your Mac to diagnose Arial + cropping issues.

Usage: uv run python diagnose.py
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, findfont, FontProperties
import publiplots as pp


print("=" * 60)
print("publiplots diagnostic")
print("=" * 60)
print(f"publiplots location: {pp.__file__}")
print(f"matplotlib backend: {matplotlib.get_backend()}")
print()

print("--- Font check ---")
print(f"font.family     = {plt.rcParams['font.family']}")
print(f"font.sans-serif = {plt.rcParams['font.sans-serif'][:3]}")

arial_entries = [f for f in fontManager.ttflist if "Arial" in f.name]
print(f"Arial fonts registered: {len(arial_entries)}")
for f in arial_entries[:3]:
    print(f"  {f.name} <- {f.fname}")

resolved = findfont(FontProperties(family=plt.rcParams['font.family']))
print(f"Resolved font: {resolved}")
print()

print("--- rcParams check ---")
print(f"axes.linewidth          = {plt.rcParams['axes.linewidth']}")
print(f"xtick.major.width       = {plt.rcParams['xtick.major.width']}")
print(f"subplots.axes_size      = {pp.rcParams['subplots.axes_size']}")
print()

print("--- Cache check ---")
from pathlib import Path
import os
cache_home = os.environ.get("MPLCONFIGDIR") or str(Path.home() / ".matplotlib")
cache_dirs = [
    Path.home() / ".matplotlib",
    Path.home() / ".cache" / "matplotlib",
    Path.home() / "Library" / "Caches" / "matplotlib",  # macOS
]
for d in cache_dirs:
    if d.exists():
        for f in d.glob("fontlist-*.json"):
            print(f"  stale cache? {f}")
        print(f"  {d} exists")
print()

print("If Arial isn't resolved above, try:")
print("  rm ~/.matplotlib/fontlist-*.json  # or the path shown above")
print("  # then re-run this script")
