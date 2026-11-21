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
  - `Shot`, `Sequence`, and `Project` store independent setups and named
    sequences. Project JSON stores external background references relative
    to the project location where possible.
  - `ShotRenderer` shares seeded crowd/dog composition and held-background
    caching between animation export, frame-zero cards, and sequence export.
  - `App` serializes loading, thumbnails, previews, and exports through
    one spawned worker process. Only the Tk thread creates `ImageTk.PhotoImage`
    objects, updates controls, or opens dialogs. Job results are polled with
    `after`; stale project/shot/preview results must not be applied.
    Worker functions/arguments must be picklable and contain no Tk objects.
    Animated preview reuses a worker-local `ShotRenderer`. Project changes
    explicitly clear the worker's render caches before requesting previews.
  - `main()` starts the GUI. There is no CLI argument parser.
- `crowd.py`
  - Loads and caches layered walk-cycle assets from `Animations/`.
  - `make_crowd` creates a reproducible crowd from `Params.crowd_seed`;
    `draw_frame` composites sprites in depth order.
  - `_PersonFrames` lazily recolors and caches at most 12 sprites per person.
    Preserve frame-zero route bounds, RNG consumption, and exact sprite pixels.
  - `WalkRoute`, `Obstacle`, and `Person` represent routes, avoidance, and
    walker motion.
  - `DogInteraction` resolves the optional custom-pedestrian approach, pet,
    reversed return, and distance-based dog following. Its ground offset
    changes the real stop; all render paths share the same choreography.
    `InteractionObstacles` caches deterministic passing sides from frame zero
    to prevent crowd walkers jumping across moving actors when scrubbing.
    Ordinary shots retain the original seeded crowd and obstacle path.
  - Built-in `petting` layers have a male-hair-only fallback and use standing
    body scale, not a crop enlarged to standing height. Generic imported clips
    still require both hair variants.
- `dog.py`
  - Loads and caches dog cycles from `Animations/dog_forward/` and
    `Animations/dog_backward/`.
  - `Dog` provides the center obstacle and a gaze-directed animated sprite.
    During an interaction, scene composition uses the dog's resolved position.
    `load_interaction_assets` composites `dog_excited` overlays from
    `dog_backwards_blink` on the source canvas and loads `dog_walk` separately,
    preserving sitting-body scale and paw registration.
- `BG.png` is the default road texture, loaded relative to
  `crosswalk_generator.py`. Users can select other images in the Crosswalk tab.
  Do not assume the current working directory is the asset root.

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
  drawings. Multiple selected background images also advance using `dog_hold`,
  including when stripe wobble is zero.
- Use `render(params, width=None, height=None, supersample=3, frame=0)` for
  still background rendering. It returns a Pillow image; the GUI separately
  composites the crowd and dog.
- GUI exports are user-selected: PNG export saves one frame; animation export
  writes numbered PNGs next to a GIF. Sequence export streams PNGs in the
  worker, using `Sequence1_shot1.0001.png` naming and each saved shot's
  dimensions and duration. Frame numbering restarts per shot; PNGs do not
  encode FPS. Existing sequence-prefix frames block export, not get overwritten.
  Avoid tests or experiments that write
  into `Renders/` unless explicitly intended.
- Worker progress is bounded, coalesced, and tagged by job; never enqueue full-resolution
  animation frames for the UI. Cancellation is cooperative: restore controls
  only after worker acknowledgment, not when the cancel button is pressed.
  GIF encoding and an in-flight frame/write cannot be interrupted safely.
  Final job results include authoritative progress and explicit failures.
- Project validation precedes replacement and unsaved-change decisions.
  Cards are built in small Tk batches and thumbnails populate progressively.
  Source asset caches are read-only after construction; supersampling,
  perspective interpolation/rounding, and seeded output remain unchanged.
- Editor changes do not mutate saved shots; only explicit Update shot does.
  Edit/drag targets use shot IDs, not positional indices. Project save keeps
  uncommitted editor settings separate from shot settings.
- One optional dog interaction is allowed per shot. Petting duration and finite
  ground-position offsets persist with each custom pedestrian. Preview drag
  and nudge controls project screen deltas onto the ground, leaving authored
  route waypoints unchanged. Positioning pins the Approach duration; generated
  phases follow it. Fit sequence explicitly rounds completion up to 12 frames.
  Clips cannot overlap the reserved petting interval, and the dog must be shown.

## Development and verification

Run the app from the workspace root with:

```powershell
python crosswalk_generator.py
```

The runtime packages are Pillow and NumPy; Tkinter must be available from the
Python installation. No requirements file or project build configuration is
present. Run the focused tests with:

```powershell
python -m unittest test_backgrounds test_sequences
```

Tests use temporary asset/output files and withdrawn Tk roots. For source-only
syntax validation, run:

```powershell
python -m py_compile crosswalk_generator.py test_sequences.py
```

For behavior changes, prefer a focused check of the affected rendering or
motion behavior, then launch the GUI to verify the relevant controls and
export path. Keep documentation aligned with the actual GUI and asset layout.
