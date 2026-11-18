# Agent guide

## Project at a glance

This is a Python 3.10+ Tkinter/Pillow/NumPy desktop tool for rendering
stylized crosswalk scenes. The workspace also contains substantial hand-drawn
source art, animation files, references, and generated renders. Keep changes
focused on the requested task; do not rename, regenerate, or remove art and
render assets unless asked.

## Source map

- `crosswalk_generator.py`
  - `Params` is the shared scene configuration.
  - `Camera` handles ground-to-camera coordinates and perspective projection.
  - `_stripe_runs`, `_sketch_stripe`, and `render` build the crosswalk and
    textured background.
  - `App` owns the Tkinter controls, preview, animation, and save workflows.
  - `main()` starts the GUI. There is no CLI argument parser.
- `crowd.py`
  - Loads and caches layered walk-cycle assets from `Animations/`.
  - `make_crowd` creates a reproducible crowd from `Params.crowd_seed`;
    `draw_frame` composites sprites in depth order.
  - `WalkRoute`, `Obstacle`, and `Person` represent routes, avoidance, and
    walker motion.
- `dog.py`
  - Loads and caches dog cycles from `Animations/dog_forward/` and
    `Animations/dog_backward/`.
  - `Dog` provides the center obstacle and a gaze-directed animated sprite.
- `BG.png` is loaded relative to `crosswalk_generator.py`. Do not assume the
  current working directory is the asset root.

## Important behavior and invariants

- Keep the required asset paths and filename patterns intact. Pedestrian
  cycles use 12 frames; the dog loader discovers PNG frames in its two dog
  animation directories.
- Crowd appearance and placement are seeded. Preserve seed-based
  reproducibility when changing randomization.
- Walkers use camera-relative forward/backward animation, enter and exit
  outside the image, and are composited far-to-near. The dog can affect walker
  routes even when rendering animation frames.
- Stripe wobble and road-texture motion share a drawing index derived from
  `dog_hold`; changing that relationship can desynchronize the intended held
  drawings.
- Use `render(params, width=None, height=None, supersample=3, frame=0)` for
  still background rendering. It returns a Pillow image; the GUI separately
  composites the crowd and dog.
- GUI exports are user-selected: PNG export saves one frame; animation export
  writes numbered PNGs next to a GIF. Avoid tests or experiments that write
  into `Renders/` unless explicitly intended.

## Development and verification

Run the app from the workspace root with:

```powershell
python crosswalk_generator.py
```

The runtime packages are Pillow and NumPy; Tkinter must be available from the
Python installation. No requirements file, project build configuration, or
automated test suite is present. For source-only syntax validation, run:

```powershell
python -m py_compile crosswalk_generator.py crowd.py dog.py
```

For behavior changes, prefer a focused check of the affected rendering or
motion behavior, then launch the GUI to verify the relevant controls and
export path. Keep documentation aligned with the actual GUI and asset layout.
