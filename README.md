# CrossRoads

CrossRoads is a small Python/Tkinter art tool for creating a stylized,
three-quarter-view crosswalk scene. It combines hand-drawn, crayon-textured
stripes and road texture with animated pedestrians and a dog. Use the GUI to
adjust the scene, preview motion, and save a still image or an animation.

## Run it

You need Python 3.10 or newer, Pillow, and NumPy. Tkinter must also be
available in your Python installation (it is included with most standard
Windows Python installers).

Install the Python packages if needed:

```powershell
python -m pip install Pillow numpy
```

From the workspace root, launch the application:

```powershell
python crosswalk_generator.py
```

The Crosswalk, Crowd, and Dog tabs control the scene. When **Animate preview**
is off, the preview shows the first frame of the animation export. **Save PNG…**
exports the displayed frame at the selected output dimensions. **Save
Animation…** exports numbered PNG frames and a GIF; animations containing
pedestrians play once so walkers do not jump back to their starting positions.
In the Crowd tab, **Show character shadows** applies to both pedestrians and
the dog. Choose **Contact** for short shadows or **Cast** for longer shadows
in a fixed direction. Contact shadows are small ground blobs; cast shadows follow
each character's current drawing and retain its textured edges. Shadows vary
subtly at the **Frames per dog / road drawing** interval, even when stripe
wobble is off. They are enabled by default and appear in previews, stills,
animations, and sequence exports.
In the Crosswalk tab, **Choose backgrounds…** selects one or more images as
road textures; selecting multiple images replaces the previous selection and
cycles them in the file chooser's returned order. They advance every **Frames
per dog / road drawing** frames, even when stripe wobble is zero. **Use default**
restores `BG.png`. The preview and exports use the same sequence.
**Perspective background (match camera)** is enabled by default and projects
the texture onto the ground plane using the scene camera; turn it off for a
flat background.
Background tiles keep the source image's orientation and repeat without
mirroring, so diagonal strokes retain a consistent direction.
Use **Set defaults** to save the current scene settings for future launches;
they are stored in `~/.crossroads/defaults.json`, including selected image
paths, so keep those files available.

## Shot sequences and projects

Use the **Sequence** panel below the preview to assemble shots:

1. Set up a shot, then click **Add to sequence**. The card stores a separate
   copy of all settings and shows frame one, even if the preview is animating.
2. Drag a card by its image or details to change the order, or use its **<**
   and **>** buttons. The highlighted card indicates the drop position.
   **Remove** deletes a shot after confirmation.
3. Click **Edit** to load a shot into the scene controls. Click **Update shot**
   to apply your changes to that card, or **Done editing** to leave it.
   Changing controls alone never changes a saved shot.
4. Use **New**, **Rename**, **Delete**, and the sequence selector to manage
   multiple sequences. Names must be unique and valid Windows filename prefixes.
5. Click **Render sequence...** to export the selected sequence to a folder.
   Frames are named `Sequence1_shot1.0001.png`, `Sequence1_shot1.0002.png`,
   then `Sequence1_shot2.0001.png`, and so on. Shot numbers follow card order;
   frame numbering restarts for each shot.

Each shot keeps its own dimensions, FPS, and **Cycles** setting. Its length
is Cycles × 12 frames, even for an unchanging setup. PNG frames do not contain
playback FPS metadata; use each card's FPS when importing into an editor.
Sequence export writes PNGs only, not a combined GIF or video.
Progress appears beside the scene controls. Sequence rendering and **Save
Animation…** run in a single background worker process so the window stays responsive.
**Cancel task** stops before the next frame/write and retains completed files.
An in-progress render, write, or GIF encoding must finish before cancellation
is acknowledged; a high-resolution frame may take a while. Rendering locks
editing/export controls and pauses animated preview until the worker finishes
or acknowledges cancellation.
Choose a destination without existing PNG frames for that sequence prefix:
exports refuse to overwrite or mix in old frames. Failed renders report the
partial output location; a failed write may leave an incomplete PNG.

The **Project** menu provides **New project**, **Open project...**, **Save
project**, and **Save project as...**. A JSON project saves all named sequences,
their shots and order, the selected sequence, and the current editor setup.
Saving a project does **not** implicitly update the shot being edited: the
editor setup is saved separately. An asterisk in the title indicates unsaved
project changes; opening/new/closing prompts to save, discard, or cancel.
New projects start with the built-in scene settings.

Background images are referenced, not bundled. Paths are saved relative to
the project file where possible, so keep those images available and preserve
their relative locations when moving a project. Invalid projects or missing
backgrounds are reported without replacing your current work. **Set defaults**
still only changes startup scene settings; it does not save sequences.
Project validation runs in the background. Once a valid project is opened,
shot cards initially show **Loading preview...** and their thumbnails fill in
progressively; the editor preview is also rendered in the background. Editing
a shot or switching projects discards outdated preview results. **Cancel task**
can cancel an in-progress project load without replacing the current project.
All background jobs share one worker process rather than rendering multiple
full-resolution frames simultaneously. Animated preview also renders there,
reusing its shot renderer between frames, and waits while the worker is busy.
The first background job starts that process; later jobs reuse it.
JSON setups without a saved shadow style use cast shadows; existing shadow
settings are preserved. Opening an older project does not rewrite it: the file
is updated only when you save it. Turn shadows off in individual shots to
retain their previous appearance.

## Workspace contents

- `crosswalk_generator.py` — GUI, scene parameters, camera projection, road
  and stripe rendering, and image/animation export.
- `crowd.py` and `dog.py` — pedestrian sprites and motion, dog animation,
  obstacle avoidance, and compositing.
- `BG.png` — road texture used by the renderer.
- `Animations/` — layered walk cycles and dog frames used by the crowd and dog
  renderers. These files are required at their current paths.
- `Renders/` — rendered stills, frame sequences, GIFs, and a full render.
- `krita/`, `comp.blend`, and `Sketch_V2.bundle` — associated art and
  animation-project files.
- `reference/` — visual references; `aufio/` — audio clips.
- `crosswalk_single_0deg.png` — an example exported image.

## Notes

There is no dependency manifest. Run the focused background and sequence tests
with `python -m unittest test_backgrounds test_sequences` (Tkinter must be available). The
application is launched as a desktop GUI rather than a command-line renderer.
The tests cover pixel-equivalent rendering optimizations, real-process
loading/preview/export, UI heartbeats, cancellation, and partial-output failures.
See [agent_info.md](agent_info.md) for code structure and practical guidance
when changing the project.
