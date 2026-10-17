"""homr as a recognition engine over HTTP (SheetCheater task 0015, development builds only).

    <venv>/Scripts/python engines/homr/server.py [--port 8765] [--cache <dir>] [--dml]

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
Public License as published by the Free Software Foundation, version 3 (LICENSE in this folder). It runs homr
(https://github.com/liebharc/homr, AGPL-3.0, code and model weights) unmodified as a library, except for the one
change in patches/ (applied by setup.py), and returns what homr read as JSON. The app talks to it only over HTTP;
no part of this folder is bundled into the app.

POST /read   body: a PNG or JPEG page. Answer:
    { "engine": "homr", "version": "0.7.0", "ms": ..., "width": W, "height": H,
      "systems": [ { "staves": [ { "grand": bool, "unit": px, "grid": [[x, y0, y1, ...], ...],
                                   "heads": [[x, y, w, h], ...], "bars": [x, ...] } ] } ],
      "musicxml": "<score-partwise ...>", "partSystems": [[0, 1, ...], ...], "error": null | "..." }
    Every position is in pixels of the image as sent (homr's own crop and resize undone). The MusicXML is the page as
    homr's own writer makes it: one part per stave of a system (a grand staff is one part with two staves), a
    <print new-system="yes"/> at each system break; partSystems[p] says which system each of part p's systems is.
GET /health  { "engine": "homr", "version": ..., "ok": true }
"""

import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import onnxruntime as ort

ARGS = sys.argv[1:]


def _arg(name: str, default: str | None = None) -> str | None:
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


# --dml (a venv with onnxruntime-directml): the models named in HOMR_DML (default segnet, encoder) ask for DirectML
# first; homr itself only knows CUDA and CoreML. Output is the same as on the CPU (measured in task 0014).
if "--dml" in ARGS:
    _Session = ort.InferenceSession
    _DML = os.environ.get("HOMR_DML", "segnet,encoder").split(",")

    def _dml_session(path, sess_options=None, providers=None, **kw):  # type: ignore[no-untyped-def]
        on_gpu = isinstance(path, str) and any(p and p in os.path.basename(path) for p in _DML)
        want = ["DmlExecutionProvider", "CPUExecutionProvider"] if on_gpu else (providers or ["CPUExecutionProvider"])
        return _Session(path, sess_options, providers=want, **kw)

    ort.InferenceSession = _dml_session  # type: ignore[misc]

import homr.main as hm  # noqa: E402
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml  # noqa: E402
from homr.staff_parsing import _ensure_same_number_of_staffs, parse_staff_image  # noqa: E402
from homr.staff_regions import StaffRegions  # noqa: E402
from homr.transformer.configs import Config  # noqa: E402
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols  # noqa: E402

VERSION = "0.7.0"
# The answer's shape; part of the cache key.
FORMAT = 3
PORT = int(_arg("--port", "8765") or 8765)
CACHE = _arg("--cache")

# homr crops the page to its paper and resizes it to 1920 px wide before it reads; the crop is recorded here so
# every position can be given back in the frame of the image as sent.
_crop: dict[str, int] = {}
_autocrop = hm.autocrop


def _recording_autocrop(img):  # type: ignore[no-untyped-def]
    out = _autocrop(img)
    if out is img or out.shape == img.shape:
        _crop.update(x=0, y=0, w=img.shape[1], h=img.shape[0])
    else:
        # A crop is a view of the image: its offset in bytes gives its corner.
        offset = out.__array_interface__["data"][0] - img.__array_interface__["data"][0]
        _crop.update(x=(offset % img.strides[0]) // img.strides[1], y=offset // img.strides[0], w=out.shape[1], h=out.shape[0])
    return out


hm.autocrop = _recording_autocrop


def read_page(path: str, width: int, height: int) -> dict:
    config = hm.ProcessingConfig(False, False, False, False, -1, False, False, False)
    multi_staffs, image, debug, title_future = hm.detect_staffs_in_image(path, config)
    frame_h, frame_w = image.shape[:2]
    sx = _crop["w"] / frame_w
    sy = _crop["h"] / frame_h

    def px(x: float, y: float) -> tuple[float, float]:
        return (round(float(x) * sx + _crop["x"], 1), round(float(y) * sy + _crop["y"], 1))

    transformer = Config()
    # As homr's own command line on the CPU: fp32 models (its fp16 ones are only for CUDA).
    transformer.use_gpu_inference = False
    transformer.use_coreml_encoder = False
    staffs = _ensure_same_number_of_staffs(multi_staffs, image)
    regions = StaffRegions(staffs)
    systems = []
    # Tokens per system and stave of the system, read in homr's own order (stave by stave, voice-major).
    tokens: list[list[list[EncodedSymbol]]] = [[[] for _ in multi.staffs] for multi in staffs]
    index = 0
    for voice in range(len(staffs[0].staffs) if staffs else 0):
        for sys_no, multi in enumerate(staffs):
            if voice < len(multi.staffs):
                try:
                    tokens[sys_no][voice] = parse_staff_image(debug, index, multi.staffs[voice], image, regions, transformer)
                except Exception:  # noqa: BLE001 - one stave homr cannot crop (a zero-width crop) costs that stave only
                    traceback.print_exc()
            index += 1
    for sys_no, multi in enumerate(staffs):
        staves = []
        for staff in multi.staffs:
            grid = []
            for p in staff.grid[:: max(1, len(staff.grid) // 24)] + [staff.grid[-1]]:
                pts = [px(p.x, y) for y in p.y]
                grid.append([pts[0][0], *[q[1] for q in pts]])
            heads = []
            for n in staff.get_notes():
                cx, cy = px(*n.box.center)
                heads.append([cx, cy, round(float(n.box.size[0]) * sx, 1), round(float(n.box.size[1]) * sy, 1)])
            bars = sorted(px(b.center[0], b.center[1])[0] for b in staff.get_bar_lines())
            staves.append({"grand": bool(staff.is_grandstaff), "unit": round(float(staff.average_unit_size) * sy, 2), "grid": grid, "heads": heads, "bars": bars})
        systems.append({"staves": staves})
    # One MusicXML for the page as homr writes it (parse_staffs: a voice is the same stave of every system, a new
    # system after each stave read; empty staves skipped), with which system each of a part's system breaks is.
    voices: list[list[EncodedSymbol]] = []
    part_systems: list[list[int]] = []
    for voice in range(len(staffs[0].staffs) if staffs else 0):
        joined: list[EncodedSymbol] = []
        seen: list[int] = []
        for sys_no in range(len(staffs)):
            t = tokens[sys_no][voice] if voice < len(tokens[sys_no]) else []
            if not t:
                continue
            joined.extend(t)
            joined.append(EncodedSymbol("newline"))
            seen.append(sys_no)
        voices.append(remove_duplicated_symbols(joined))
        part_systems.append(seen)
    xml = generate_xml(XmlGeneratorArguments(False, None, None), voices, "")
    with tempfile.NamedTemporaryFile("w+", suffix=".musicxml", delete=False) as f:
        out = f.name
    try:
        xml.write(out)
        with open(out, encoding="utf-8") as f:
            musicxml = f.read()
    finally:
        os.remove(out)
    try:
        title_future.result(60)
    except Exception:  # noqa: BLE001 - the title is not used
        pass
    debug.clean_debug_files_from_previous_runs()
    return {"engine": "homr", "version": VERSION, "format": FORMAT, "width": width, "height": height, "systems": systems, "musicxml": musicxml, "partSystems": part_systems, "error": None}


def answer(body: bytes) -> dict:
    key = hashlib.sha1(body + str(FORMAT).encode()).hexdigest()
    cached = os.path.join(CACHE, key + ".json") if CACHE else None
    if cached and os.path.exists(cached):
        with open(cached, encoding="utf-8") as f:
            return json.load(f)
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"engine": "homr", "version": VERSION, "systems": [], "error": "The image could not be decoded."}
    t = time.time()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "page.png")
        cv2.imwrite(path, img)
        try:
            result = read_page(path, img.shape[1], img.shape[0])
        except Exception as e:  # noqa: BLE001 - homr's own failures ("No staffs found") are an answer
            traceback.print_exc()
            result = {"engine": "homr", "version": VERSION, "width": img.shape[1], "height": img.shape[0], "systems": [], "error": f"{type(e).__name__}: {str(e)[:300]}"}
    result["ms"] = round((time.time() - t) * 1000)
    # Only answers are kept; a failure is tried again next time.
    if cached and result["error"] is None:
        os.makedirs(CACHE, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as f:
            json.dump(result, f)
    return result


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # The app's dev server is another origin, and the exported build is cross-origin isolated.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server's name
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, {"engine": "homr", "version": VERSION, "ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/read"):
            self._send(404, {"error": "not found"})
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._send(200, answer(body))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - http.server's name
        sys.stderr.write("homr-engine: " + (format % args) + "\n")


if __name__ == "__main__":
    hm.download_weights(False, False, False)
    print(f"homr engine {VERSION} on http://127.0.0.1:{PORT}", file=sys.stderr, flush=True)
    # One page at a time: homr's models are loaded once and are not shared between threads.
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
