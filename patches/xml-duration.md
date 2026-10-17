# The one change to homr 0.7.0

File: `homr/music_xml_generator.py` (applied by `../setup.py`, idempotent).

Three places write `<duration>` as `int(fraction * division)`, which is 0 for a
duration under one division; MusicXML readers refuse such a file and the whole
page is lost (3 of 180 pages in task 0014). Each becomes `max(1, int(...))`:

```diff
-        note.add_child(mxl.XMLDuration(value_=int(model_duration.fraction * state.division)))
+        note.add_child(mxl.XMLDuration(value_=max(1, int(model_duration.fraction * state.division))))
-            backup.add_child(mxl.XMLDuration(value_=int(group_duration * state.division)))
+            backup.add_child(mxl.XMLDuration(value_=max(1, int(group_duration * state.division))))
-            mxl.XMLDuration(value_=int((final_duration - chord_duration) * state.division))
+            mxl.XMLDuration(value_=max(1, int((final_duration - chord_duration) * state.division)))
```

Licensed under the AGPL-3.0 like homr itself (`../LICENSE`).
