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

The Crosswalk, Crowd, Dog, and Custom Pedestrians tabs control the scene. When
**Animate preview** is off, the preview shows the first frame unless you are
scrubbing the Custom Pedestrians timeline. **Save PNG…**
exports the displayed frame at the selected output dimensions. **Save
Animation…** exports numbered PNG frames and a GIF; animations containing
pedestrians play once so walkers do not jump back to their starting positions.
In the Crowd tab, **Loop pedestrians** is on by default. Turn it off to send a
new randomly styled pedestrian in when a crowd walker exits the scene.
In the Dog tab, **Dog blinks occasionally** is on by default and adds brief,
irregular blinks when the dog faces the camera. Turn it off to keep its eyes
open. Blink timing is consistent across previews and exports; short animations
may end before the first blink, and looping GIFs repeat the captured pattern.
The dog watches nearby automatic, custom, and imported pedestrians using the
same smoothed gaze. Dog-petting and following animations take precedence during
their interaction.
In the Crowd tab, **Show character shadows** applies to both pedestrians and
the dog. Choose **Contact** for short shadows or **Cast** for longer shadows
in a fixed direction. Contact shadows are small ground blobs; cast shadows follow
each character's current drawing, meet its feet, and retain its textured edges. Shadows vary
subtly at the **Frames per dog / road drawing** interval, even when stripe
wobble is off. They are enabled by default and appear in previews, stills,
animations, and sequence exports.
In the Crosswalk tab, **Camera up/down (%)** moves the camera vertically
relative to its view without changing **Camera tilt**. Increase it to move the
dog lower in the frame, or decrease it to move the dog higher. The shift is a
percentage of the unshifted view width, and `0` keeps the original framing.
This setting is saved per shot and applies to previews and exports.

In the Crosswalk tab, **Choose backgrounds…** selects one or more images as
road textures; selecting multiple images replaces the previous selection and
cycles them in the file chooser's returned order. With background animation
enabled, they advance every **Frames per dog / road drawing** frames, even
when stripe wobble is zero. **Use default**
restores `BG.png`. The preview and exports use the same sequence.
**Animate background** and **Animate stripes** are on by default and can be
controlled independently. Turning off background animation holds the first
road image and texture position throughout the shot; turning off stripe
animation holds the first stripe drawing. The wobble slider controls motion
strength for both when enabled. Characters and their shadows still animate.
**Perspective background (match camera)** is enabled by default and projects
the texture onto the ground plane using the scene camera; turn it off for a
flat background.
Background tiles keep the source image's orientation and repeat without
mirroring, so diagonal strokes retain a consistent direction.
Use **Set defaults** to save the current scene settings for future launches;
they are stored in `~/.crossroads/defaults.json`, including selected image
paths, so keep those files available.

## Custom pedestrians

The **Custom Pedestrians** tab starts empty. It is independent of the automatic
crowd and works even when the automatic crowd is disabled.

1. Click **Add pedestrian**. Pick colors for the legs, arms, head, and hair,
   choose a hair variant, and set **Speed** in world units per second. Use
   **Size…** to set that pedestrian's height in world units. Until you set an
   individual height, the pedestrian uses the Crowd tab's **Person size**.
2. Click **Draw route / place**. In the shot preview, click the start, any waypoints,
   and the stop, then click **Finish route**. Use **Undo point** to correct your
   last click. Drawing a new route replaces the old one only when you finish it.
3. Use the **Scrub** row to preview a frame. In the **Walk** lane, drag the
   middle of the block to change when walking starts, or drag an edge to
   resize it. Shorter blocks walk faster and longer blocks walk slower, so
   the pedestrian still completes the full route. They wait at the start
   before the block and stay at the stop afterward.
4. Optionally use **Add clip** to select a layered PNG folder and set an
   inclusive frame range. Drag or resize clips in the **Animation** lane,
   before, after, or over the Walk block. Clips can't overlap each other on
   that lane. They change the drawing, not the movement: during the Walk
   block the pedestrian keeps moving, and outside it the clip plays in place.
   The clip uses one drawing per shot frame and repeats if its block is longer
   than the source. Outside clips, the built-in drawings are used.

Each clip folder needs `legs`, `arms`, `head`, `hair_male`, and `hair_female`
subfolders like [the built-in walk cycle](./Animations/walk_forward). Each
subfolder needs the same consecutive frame numbers at the end of its PNG
filenames (for example, `walk_arms0001.png`). All layers must share the same
canvas size; use transparency to align the parts. Clips can have any positive
number of frames. Folder references are saved with the shot, not copied. If a
folder moves, select its block and use **Edit / relink**. Custom pedestrians
are saved in projects, not in startup defaults.

### Imported side-view walkers

Click **Add imported…** and choose a folder of consecutive numbered PNG
frames, such as `Animations/oldman_walk`. Export the frames from Krita first;
`.kra` files cannot be imported directly. Frames must share a canvas size and
include transparency. Their original colors are preserved. The artwork should
face left: it flips automatically for rightward travel in the camera view.
On a vertical segment it keeps the preceding horizontal facing, or faces left
if there is none.

To animate an imported character in place, click **Draw route / place**, click
its position in the preview, and click **Finish route**. The animation repeats
from the first shot frame, advancing one drawing per shot frame, in the artwork's
original orientation. Use **Size…** to adjust it. To reposition it, repeat those
steps; to make it walk instead, add at least two route points.

For walking, draw a route and use the same speed, size, and Walk timeline controls as other
custom pedestrians. **Stride…** sets the distance traveled per animation cycle,
in pedestrian heights. The cycle follows distance, so its effective frame rate
changes with speed; the first drawing is held when not walking. Imported
walkers cannot use the built-in color controls, layered clips, or dog-petting
drawings. Use **Relink walk…** if the folder moves. Project files reference the
folder rather than copying the artwork, and save its path relative to the
project where possible. Missing folders are reported and block export.

### Petting the dog

Select a custom pedestrian, draw the entry route/waypoints, enable **Show dog**,
then select **Pet dog, then leave together**. Only one pedestrian can interact
with the dog per shot. The final approach to the dog and the reversed return
route are generated automatically. The entry leg extends outside the image so
the pedestrian enters naturally and both characters can leave completely.
The Walk block becomes **Approach**: its duration covers the resolved approach,
including the off-screen extension and final petting stop. Resize it to adjust
walking speed. Set **Pet seconds** and click **Apply** to repeat the petting
drawings for that duration. Generated **Pet** and **Return + dog** blocks show
the rest of the sequence; imported clips cannot overlap the petting interval.

Use **Position petting pose** to pause at the petting frame. The preview
temporarily zooms in on the dog (up to 4x) and hides route lines, dots, and labels.
Drag in the preview to align the pedestrian's hand with the dog, or use
**Left / Right / Up / Down**
with the **Nudge (preview px)** step for fine adjustments. **Reset** restores
automatic placement; **Done positioning** restores the normal view and route
guides. The close-up is preview-only and does not change the shot camera or
exports. Positioning moves the actual petting stop, not just its drawing,
so the approach and return remain
connected. Positioning preserves the Approach block's duration. The offset is
saved per pedestrian in ground coordinates and does not depend on export size.

The dog receives the excited overlay while being petted, then follows the return
path until fully off-screen. Automatic crowd walkers steer around the pedestrian
and moving dog; this is local steering, not general crowd collision simulation.
Shadows and normal depth layering follow the adjusted positions.
The timeline warns if the shot cuts off the interaction. **Fit sequence**
explicitly increases Cycles enough to include the dog's exit; shot length never
grows silently.

The built-in petting cycle currently uses male hair for both hair variants.
Female walking hair is unchanged. The supplied dog-walk art only shows the back
view, which is used even when moving toward the camera. Interaction drawings use
the Dog tab's **Frames per dog / road drawing** hold. Shots without the interaction
keep their existing behavior.

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
is Cycles × 12 frames, even for an unchanging setup. The Cycles spinner goes
up to 500 (6,000 frames). PNG frames do not contain
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
Older projects and saved defaults without a blink setting enable blinking;
the toggle is saved separately for the editor and each shot.
Older projects and saved defaults preserve their combined background/stripe
animation setting when opened. Both toggles are saved separately for the
editor and each shot.

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
