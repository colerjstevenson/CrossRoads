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

The Crosswalk, Crowd, and Dog tabs control the scene. **Save PNG…** exports the
displayed animation frame at the selected output dimensions. **Save
Animation…** exports numbered PNG frames and a GIF; animations containing
pedestrians play once so walkers do not jump back to their starting positions.

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

There is no dependency manifest or automated test suite in the workspace.
The application is launched as a desktop GUI rather than a command-line
renderer. See [agent_info.md](agent_info.md) for code structure and practical
guidance when changing the project.
