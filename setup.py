"""Applies this folder's one change to an installed homr 0.7.0 (patches/xml-duration.md).

    <venv>/Scripts/python engines/homr/setup.py

homr's MusicXML writer can write a duration of 0 (int() of a fraction under one division); MusicXML readers refuse
the file and the page is lost (3 of 180 pages in task 0014). Every such duration becomes at least 1. Idempotent.
"""

import importlib.util
import os
import sys

spec = importlib.util.find_spec("homr.music_xml_generator")
if spec is None or spec.origin is None:
    sys.exit("homr is not installed in this Python (pip install -r engines/homr/requirements.txt)")
path = spec.origin
src = open(path, encoding="utf-8").read()
CHANGES = [
    ("mxl.XMLDuration(value_=int(model_duration.fraction * state.division))", "mxl.XMLDuration(value_=max(1, int(model_duration.fraction * state.division)))"),
    ("mxl.XMLDuration(value_=int(group_duration * state.division))", "mxl.XMLDuration(value_=max(1, int(group_duration * state.division)))"),
    ("mxl.XMLDuration(value_=int((final_duration - chord_duration) * state.division))", "mxl.XMLDuration(value_=max(1, int((final_duration - chord_duration) * state.division)))"),
]
done = 0
for old, new in CHANGES:
    if new in src:
        done += 1
    elif old in src:
        src = src.replace(old, new)
        done += 1
if done != len(CHANGES):
    sys.exit(f"{path}: expected homr 0.7.0's writer; {done} of {len(CHANGES)} places found")
open(path, "w", encoding="utf-8").write(src)
print(f"{os.path.basename(path)}: durations of at least 1 ({path})")
