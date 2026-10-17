# homr engine (development only)

[homr](https://github.com/liebharc/homr) 0.7.0 as a separate program the app
talks to over HTTP (task 0015). homr's code and model weights are licensed
under the GNU Affero General Public License v3 (`LICENSE`); so is everything
in this folder. Nothing in this folder is part of the app or its bundle; the
app's reader (`src/features/reading/homr/`) only sends a page image to this
server and reads its JSON answer. It is enabled in development builds only
(`EXPO_PUBLIC_HOMR_URL`, see `src/config.ts`).

Changes to homr: one, `patches/xml-duration.md` (applied by `setup.py`).

## Run

All installs stay inside the repo (`.bench/` is git-ignored):

    python -m venv .bench/homr-venv
    .bench/homr-venv/Scripts/pip install -r engines/homr/requirements.txt
    .bench/homr-venv/Scripts/python engines/homr/setup.py
    .bench/homr-venv/Scripts/python engines/homr/server.py --port 8765 --cache .bench/homr-engine-cache

`--dml` puts segnet and the encoder on the GPU through DirectML (a venv with
`onnxruntime-directml` instead of `onnxruntime`): about 2-12 s a page on the
RX 9070 XT against 10-25 s on the CPU. `--cache <dir>` keeps every answer by
the image's hash.

Then start the app with `EXPO_PUBLIC_HOMR_URL=http://127.0.0.1:8765`.
Stop the server with `sh scripts/bench/stop-engine.sh`.

## Training data and weights

The transformer checkpoint used (396) was trained on OpenScore Lieder and
StringQuartets (CC0), GrandStaff and Camera-PrIMuS; Camera-PrIMuS is CC BY-NC
4.0 on its Hugging Face copy. See the task 0014 and 0015 notes before any use
beyond development.

## Where this is used

This folder is the homr-based note reader of SheetCheater
(https://sheetcheater.lima-spielhofen.com). It is published here as required
by the AGPL-3.0: the browser engine in `web/` is exactly what the live site
loads as its own worker (`web/worker.ts`), and `server.py` is the development
engine. Model weights are homr 0.7.0's (checkpoint 396) from
https://github.com/liebharc/homr, unchanged, AGPL-3.0.
