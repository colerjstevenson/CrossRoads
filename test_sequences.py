import json
import math
import queue
import tempfile
import threading
import time
import tkinter as tk
import unittest
from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageTk

import crosswalk_generator as app


class _ThreadWorker:
    """Keep delayed-job tests deterministic; integration tests use the real process."""

    def __init__(self):
        self.cancelled = threading.Event()
        self.progress = queue.Queue(maxsize=1)
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, function, token):
        return self.pool.submit(app._run_job, function, self.cancelled, self.progress, token)

    def shutdown(self, *, wait=False, cancel_futures=True):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.image = self.directory / "road.png"
        Image.new("RGB", (20, 20), "red").save(self.image)
        self.params = app.Params(width=16, height=16, crowd_enabled=False,
                                 dog_enabled=False, stripe_wobble=0, anim_cycles=1,
                                 background_images=(str(self.image),))
        self.path = self.directory / "project.json"

    def test_random_scene_stays_valid_and_preserves_user_assets(self):
        params = replace(self.params, width=640, height=480)
        generated = app.random_scene_params(params, app.random.Random(12))

        app.validate_project_params(generated)
        self.assertEqual((generated.width, generated.height), (640, 480))
        self.assertEqual(generated.background_images, params.background_images)
        self.assertEqual(generated.custom_pedestrians, params.custom_pedestrians)
        self.assertEqual(generated.camera_shift, params.camera_shift)
        for key in ("stripe_count", "stripe_width", "stripe_gap", "stripe_color",
                    "stripe_wobble", "stripes_animated"):
            self.assertEqual(getattr(generated, key), getattr(params, key))
        for key in ("dog_enabled", "dog_size", "dog_clearance", "dog_tilt",
                    "dog_blink", "dog_hold", "person_size", "size_var",
                    "crowd_speed"):
            self.assertEqual(getattr(generated, key), getattr(params, key))
        self.assertNotEqual(generated.depth, params.depth)
        self.assertNotEqual(generated, params)

    def test_round_trip_multiple_sequences_and_relative_assets(self):
        project = app.new_project(self.params)
        first = project.sequences[0]
        first.shots = [app.Shot(replace(self.params, angle=45, camera_shift=15)),
                       app.Shot(replace(self.params, anim_cycles=2, anim_fps=24))]
        second = app.Sequence("Other", [app.Shot(replace(self.params, width=32))])
        project.sequences.append(second)
        project.active_id = second.id
        app.save_project_file(project, self.path)
        values = json.loads(self.path.read_text())
        self.assertEqual(values["editor"]["background_images"], ["road.png"])
        loaded = app.load_project_file(self.path)
        self.assertEqual(loaded, project)
        self.assertEqual(loaded.sequences[0].shots[0].params.angle, 45)
        self.assertEqual(loaded.editor.angle, 0)
        self.assertIsNot(loaded.editor, loaded.sequences[0].shots[0].params)

    def test_legacy_sequences_open_without_rewriting_and_round_trip(self):
        project = app.new_project(self.params)
        project.sequences[0].shots = [
            app.Shot(replace(self.params, angle=45)),
            app.Shot(replace(self.params, anim_cycles=2)),
        ]
        project.sequences.append(app.Sequence(
            "Other", [app.Shot(replace(self.params, width=32))]))
        project.active_id = project.sequences[1].id
        values = app.project_values(project, self.directory)
        setups = [values["editor"]] + [
            shot["params"] for sequence in values["sequences"] for shot in sequence["shots"]]
        for setup in setups:
            del setup["shadows_enabled"]
            del setup["shadow_style"]
            del setup["dog_blink"]
            del setup["background_animated"]
            del setup["stripes_animated"]
            del setup["camera_shift"]
            del setup["loop_pedestrians"]
        self.path.write_text(json.dumps(values), encoding="utf-8")
        original = self.path.read_bytes()

        project.editor = replace(project.editor, shadow_style="cast")
        for sequence in project.sequences:
            for shot in sequence.shots:
                shot.params = replace(shot.params, shadow_style="cast")
        loaded = app.load_project_file(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(loaded, project)
        app.save_project_file(loaded, self.path)
        self.assertEqual(app.load_project_file(self.path), project)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        for setup in [saved["editor"]] + [
                shot["params"] for sequence in saved["sequences"] for shot in sequence["shots"]]:
            self.assertTrue(setup["shadows_enabled"])
            self.assertEqual(setup["shadow_style"], "cast")
            self.assertTrue(setup["dog_blink"])
            self.assertTrue(setup["background_animated"])
            self.assertTrue(setup["stripes_animated"])
            self.assertTrue(setup["loop_pedestrians"])

    def test_background_animation_round_trip_and_reject_non_boolean(self):
        project = app.new_project(replace(self.params, background_animated=False, stripes_animated=True))
        project.sequences[0].shots.append(app.Shot(replace(
            self.params, background_animated=True, stripes_animated=False)))
        app.save_project_file(project, self.path)
        self.assertEqual(app.load_project_file(self.path), project)
        values = json.loads(self.path.read_text(encoding="utf-8"))
        values["editor"]["background_animated"] = 1
        self.path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "background_animated"):
            app.load_project_file(self.path)

    def test_combined_animation_setting_migrates_per_setup(self):
        project = app.new_project(replace(self.params, background_animated=False))
        project.sequences[0].shots.append(app.Shot(replace(self.params, background_animated=True)))
        values = app.project_values(project, self.directory)
        del values["editor"]["stripes_animated"]
        del values["sequences"][0]["shots"][0]["params"]["stripes_animated"]
        self.path.write_text(json.dumps(values), encoding="utf-8")
        loaded = app.load_project_file(self.path)
        self.assertFalse(loaded.editor.background_animated)
        self.assertFalse(loaded.editor.stripes_animated)
        self.assertTrue(loaded.sequences[0].shots[0].params.stripes_animated)
        values["editor"]["stripes_animated"] = 1
        self.path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stripes_animated"):
            app.load_project_file(self.path)

    def test_blink_settings_round_trip_and_reject_non_boolean(self):
        project = app.new_project(replace(self.params, dog_blink=False))
        project.sequences[0].shots.append(app.Shot(replace(self.params, dog_blink=True)))
        app.save_project_file(project, self.path)
        self.assertEqual(app.load_project_file(self.path), project)
        values = json.loads(self.path.read_text(encoding="utf-8"))
        values["editor"]["dog_blink"] = 1
        self.path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dog_blink"):
            app.load_project_file(self.path)

    def test_pedestrian_loop_setting_round_trip_and_reject_non_boolean(self):
        project = app.new_project(replace(self.params, loop_pedestrians=False))
        project.sequences[0].shots.append(
            app.Shot(replace(self.params, loop_pedestrians=True)))
        app.save_project_file(project, self.path)
        self.assertEqual(app.load_project_file(self.path), project)
        values = json.loads(self.path.read_text(encoding="utf-8"))
        values["editor"]["loop_pedestrians"] = 1
        self.path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "loop_pedestrians"):
            app.load_project_file(self.path)

    def test_shadow_settings_round_trip_and_reject_invalid_style(self):
        project = app.new_project(replace(self.params, shadows_enabled=False,
                                          shadow_style="cast"))
        project.sequences[0].shots.append(app.Shot(replace(project.editor)))
        app.save_project_file(project, self.path)
        self.assertEqual(app.load_project_file(self.path), project)
        values = json.loads(self.path.read_text(encoding="utf-8"))
        values["sequences"][0]["shots"][0]["params"]["shadow_style"] = "invalid"
        self.path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shadow style"):
            app.load_project_file(self.path)

    def test_failed_atomic_save_preserves_previous_project(self):
        project = app.new_project(self.params)
        app.save_project_file(project, self.path)
        previous = self.path.read_bytes()
        project.editor.angle = 40
        with patch.object(app.os, "replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                app.save_project_file(project, self.path)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_rejects_malformed_projects_and_missing_backgrounds(self):
        project = app.new_project(self.params)
        project.sequences[0].shots.append(app.Shot(replace(self.params)))
        values = app.project_values(project, self.directory)
        bad_values = [
            {**values, "version": 2},
            {**values, "version": True},
            {**values, "sequences": []},
            {**values, "active_sequence": "missing"},
            {**values, "editor": {**values["editor"], "width": 0}},
            {**values, "editor": {**values["editor"], "zoom": float("nan")}},
            {**values, "editor": {**values["editor"], "background_images": [1]}},
            {**values, "editor": {key: value for key, value in values["editor"].items()
                                  if key != "angle"}},
            {**values, "editor": {}},
        ]
        duplicate = json.loads(json.dumps(values))
        duplicate["sequences"].append(duplicate["sequences"][0])
        bad_values.append(duplicate)
        for bad in bad_values:
            with self.subTest(bad=bad):
                self.path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    app.load_project_file(self.path)
        self.path.write_text(json.dumps(values))
        self.image.unlink()
        with self.assertRaisesRegex(ValueError, "Cannot read background"):
            app.load_project_file(self.path)

    def test_names_are_safe_filename_prefixes(self):
        for name in ("", " ", "../bad", "one/two", "bad*", "con", "LPT1", "last.",
                     " leading", "trailing ", "new\nline"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                app.validate_sequence_name(name)
        self.assertEqual(app.validate_sequence_name("Sequence 2"), "Sequence 2")


class CustomPedestrianTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / "gesture"
        for layer in app.crowd.BODY_LAYERS + app.crowd.HAIR_LAYERS:
            (self.folder / layer).mkdir(parents=True)
            for index in (3, 4):
                image = Image.new("RGBA", (16, 20))
                ImageDraw.Draw(image).rectangle((4 + index, 2, 12, 16), fill="white")
                image.save(self.folder / layer / f"gesture_{layer}{index:04d}.png")
        self.spec = app.crowd.CustomPedestrian(
            "walker", ((0, 0), (1, 0), (1, 2)), speed=2, departure=2,
            clips=(app.crowd.ClipBlock(str(self.folder), 0, 1),
                   app.crowd.ClipBlock(str(self.folder), 10, 11)))
        self.params = app.Params(width=64, height=64, zoom=5, anim_fps=4, anim_cycles=1,
                                 crowd_enabled=False, dog_enabled=False,
                                 custom_pedestrians=(self.spec,))
        self.addCleanup(app.crowd.load_clip.cache_clear)

    def test_ground_projection_and_constant_speed_around_corner(self):
        camera = app.Camera(replace(self.params, angle=37, tilt=55), 470, 240)
        for x, y in ((-1, 1), (0, 0), (1, 2)):
            world = camera.ground_point(*camera.project(camera.to_cam(x, y)))
            np.testing.assert_allclose(world, (x, y), atol=1e-10)
        walker = app.crowd.CustomWalker(self.spec, 1.75, 4)
        self.assertEqual([walker.pose(frame)[:2] for frame in (0, 2, 3, 4, 5, 8, 11)],
                         [(0, 0), (0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 2), (1, 2)])

    def test_imported_drawings_and_export_match_preview(self):
        walker = app.crowd.CustomWalker(self.spec, 1.75, 4)
        camera = app.Camera(self.params, 64, 64)
        def drawing(frame):
            _, _, dx, dy, distance = walker.pose(frame)
            return walker.sprite(camera, frame, dx, dy, distance)[0].tobytes()
        self.assertNotEqual(drawing(0), drawing(1))
        self.assertEqual(drawing(0), drawing(10))
        self.assertNotEqual(drawing(8), drawing(10))
        sequence = app.Sequence("Custom", [app.Shot(self.params)])
        outputs = list(app.sequence_outputs(sequence, self.folder.parent))
        for frame in (0, 5, 11):
            background, preview = app._preview_images(self.params, 64, 64, frame)
            renderer = app.ShotRenderer(self.params, supersample=2)
            self.assertEqual(preview.tobytes(), renderer.frame(frame).tobytes())
            self.assertNotEqual(preview.tobytes(), background.tobytes())
            self.assertEqual(outputs[frame][1].tobytes(), app.ShotRenderer(self.params).frame(frame).tobytes())

    def test_walk_block_completes_route_and_uses_clip_while_moving(self):
        clip = replace(self.spec.clips[0], start=3, end=4)
        p = replace(self.params, custom_pedestrians=(
            replace(self.spec, walk_duration=5, clips=(clip,)),))
        renderer = app.ShotRenderer(p, supersample=2)
        walker = renderer.custom[0]
        self.assertEqual(walker.pose(1)[:2], (0, 0))
        self.assertEqual(walker.pose(2)[:2], (0, 0))
        x, y, dx, dy, distance = walker.pose(4)
        self.assertEqual((x, y), (1, .5))
        self.assertEqual(walker.pose(6)[:2], (1, 2))
        self.assertEqual(walker.pose(7)[:2], (1, 2))
        sprite, _ = walker.sprite(renderer.cam, 4, dx, dy, distance)
        self.assertIs(sprite, walker.frames[clip.folder][0][1])
        _, preview = app._preview_images(p, 64, 64, 4)
        self.assertEqual(preview.tobytes(), renderer.frame(4).tobytes())

    def test_individual_size_changes_rendered_height_without_changing_route(self):
        small = replace(self.spec, height=0.65)
        tall = replace(self.spec, height=1.4)
        p = replace(self.params, shadows_enabled=False)
        background = Image.new("RGB", (p.width, p.height))
        for frame in (0, 5):  # Imported animation and built-in walk.
            with self.subTest(frame=frame):
                short_renderer = app.ShotRenderer(replace(p, custom_pedestrians=(small,)))
                tall_renderer = app.ShotRenderer(replace(p, custom_pedestrians=(tall,)))
                self.assertEqual(short_renderer.custom[0].pose(frame), tall_renderer.custom[0].pose(frame))
                short_box = short_renderer.compose(background, frame).getbbox()
                tall_box = tall_renderer.compose(background, frame).getbbox()
                self.assertGreater(tall_box[3] - tall_box[1], short_box[3] - short_box[1])

    def test_project_round_trip_legacy_defaults_and_import_validation(self):
        project = app.new_project(self.params)
        self.assertEqual(project.editor.custom_pedestrians, ())
        params = replace(self.params, person_size=2, custom_pedestrians=(
            replace(self.spec, height=1.25, walk_duration=9),))
        project.sequences[0].shots.append(app.Shot(params))
        path = self.folder.parent / "project.json"
        app.save_project_file(project, path)
        values = json.loads(path.read_text(encoding="utf-8"))
        raw = values["sequences"][0]["shots"][0]["params"]
        self.assertEqual(raw["custom_pedestrians"][0]["clips"][0]["folder"], self.folder.name)
        self.assertEqual(app.load_project_file(path).sequences[0].shots[0].params, params)
        person = raw["custom_pedestrians"][0]
        person["height"] = -1
        path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Pedestrian size"):
            app.load_project_file(path)
        person.pop("height")
        person["walk_duration"] = 1
        path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Walk block"):
            app.load_project_file(path)
        person.pop("walk_duration")
        path.write_text(json.dumps(values), encoding="utf-8")
        legacy = app.load_project_file(path).sequences[0].shots[0].params
        self.assertEqual(app.ShotRenderer(legacy).custom[0].height, 2)
        raw.pop("custom_pedestrians")
        path.write_text(json.dumps(values), encoding="utf-8")
        self.assertEqual(app.load_project_file(path).sequences[0].shots[0].params.custom_pedestrians, ())
        (self.folder / "arms" / "gesture_arms0003.png").unlink()
        with self.assertRaisesRegex(ValueError, "matching.*arms"):
            app.crowd.load_clip(str(self.folder))


class ImportedWalkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / "walk"
        self.folder.mkdir()
        for index, color in enumerate(("red", "blue", "green"), 1):
            image = Image.new("RGBA", (20, 20))
            draw = ImageDraw.Draw(image)
            draw.rectangle((4, 3, 8, 15), fill=color)
            draw.rectangle((9, 12, 14, 15), fill="yellow")
            image.save(self.folder / f"walk{index:04d}.png")
        self.spec = app.crowd.CustomPedestrian(
            "imported", ((0, 0), (2, 0), (2, 2), (0, 2)), departure=1,
            walk_duration=9, side_walk_folder=str(self.folder), side_walk_stride=1.0)
        self.params = app.Params(width=64, height=64, zoom=5, anim_fps=4, anim_cycles=1,
                                 crowd_enabled=False, dog_enabled=False,
                                 custom_pedestrians=(self.spec,))
        self.addCleanup(app.crowd.load_side_walk.cache_clear)

    def test_colored_frames_mirroring_and_stride(self):
        variants, anchor = app.crowd.load_side_walk(str(self.folder))
        self.assertEqual(len(variants[0]), 3)
        self.assertEqual(variants[0][0].getpixel((20, 100))[:3], (255, 0, 0))
        self.assertEqual(variants[1][0].tobytes(),
                         variants[0][0].transpose(Image.FLIP_LEFT_RIGHT).tobytes())
        self.assertEqual(anchor, (variants[0][0].width / 2, app.crowd.WORK_HEIGHT))
        camera = app.Camera(self.params, 64, 64)
        walker = app.crowd.CustomWalker(self.spec, 1, 4)
        def sprite(frame):
            pose = walker.pose(frame)
            return walker.sprite(camera, frame, *pose[2:])[0]
        self.assertIs(sprite(1), variants[1][0])
        self.assertIs(sprite(4), variants[1][0])  # Vertical segment keeps previous facing.
        self.assertIs(sprite(7), variants[0][1])
        self.assertIs(sprite(9), variants[0][0])  # Hold the first frame after arrival.
        slower = app.crowd.CustomWalker(replace(self.spec, side_walk_stride=2), 1, 4)
        pose = slower.pose(2)
        self.assertIs(slower.sprite(camera, 2, *pose[2:])[0], variants[1][1])
        pose = walker.pose(2)
        self.assertIs(walker.sprite(camera, 2, *pose[2:])[0], variants[1][2])
        vertical = replace(self.spec, route=((0, 0), (0, 2)))
        walker = app.crowd.CustomWalker(vertical, 1, 4)
        pose = walker.pose(1)
        self.assertIs(walker.sprite(camera, 1, *pose[2:])[0], variants[0][0])
        camera = app.Camera(replace(self.params, angle=180), 64, 64)
        walker = app.crowd.CustomWalker(self.spec, 1, 4)
        pose = walker.pose(1)
        self.assertIs(walker.sprite(camera, 1, *pose[2:])[0], variants[0][0])

    def test_imported_character_animates_in_place_and_survives_save(self):
        spec = replace(self.spec, route=((0, 0),), departure=5, walk_duration=None)
        params = replace(self.params, custom_pedestrians=(spec,), dog_enabled=True,
                         shadows_enabled=True, background_animated=False, stripes_animated=False)
        variants, _ = app.crowd.load_side_walk(str(self.folder))
        camera = app.Camera(params, 64, 64)
        walker = app.crowd.CustomWalker(spec, 1, params.anim_fps)
        for frame in (0, 1, 2, 3, 10):
            pose = walker.pose(frame)
            self.assertEqual(pose[:2], (0, 0))
            self.assertIs(walker.sprite(camera, frame, *pose[2:])[0], variants[0][frame % 3])
        project = app.new_project(params)
        project.sequences[0].shots.append(app.Shot(params))
        path = self.folder.parent / "stationary.json"
        app.save_project_file(project, path)
        loaded = app.load_project_file(path)
        self.assertEqual(loaded.sequences[0].shots[0].params, params)
        rendered = [app.ShotRenderer(params, supersample=2).frame(i) for i in range(3)]
        self.assertNotEqual(rendered[0].tobytes(), rendered[1].tobytes())
        _, preview = app._preview_images(params, 64, 64, 1)
        self.assertEqual(preview.tobytes(), rendered[1].tobytes())
        outputs = list(app.sequence_outputs(loaded.sequences[0], self.folder.parent))
        self.assertEqual(outputs[1][1].tobytes(), app.ShotRenderer(params).frame(1).tobytes())

        invalid = replace(spec, side_walk_folder=None)
        with self.assertRaisesRegex(ValueError, "placement point"):
            app.decode_pedestrians([asdict(invalid)])

    def test_validation_and_missing_source(self):
        app.validate_animation_folders(self.params)
        missing = replace(self.spec, side_walk_folder=str(self.folder / "missing"))
        p = replace(self.params, custom_pedestrians=(missing,))
        with self.assertRaisesRegex(ValueError, "missing.*Relink"):
            app.validate_animation_folders(p)
        background, preview = app._preview_images(p, 64, 64, 2)
        self.assertEqual(preview.tobytes(), background.tobytes())
        with self.assertRaisesRegex(ValueError, "cannot use"):
            app.validate_project_params(replace(
                self.params, custom_pedestrians=(replace(self.spec, pet_dog=True),)))
        with self.assertRaisesRegex(ValueError, "cannot use"):
            app.validate_project_params(replace(
                self.params, custom_pedestrians=(replace(
                    self.spec, clips=(app.crowd.ClipBlock("other", 0, 1),)),)))
        (self.folder / "walk0002.png").unlink()
        app.crowd.load_side_walk.cache_clear()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            app.crowd.load_side_walk(str(self.folder))
        Image.new("RGBA", (10, 10), "red").save(self.folder / "walk0002.png")
        with self.assertRaisesRegex(ValueError, "canvas size"):
            app.crowd.load_side_walk(str(self.folder))
        for index in (1, 2, 3):
            Image.new("RGBA", (20, 20)).save(self.folder / f"walk{index:04d}.png")
        with self.assertRaisesRegex(ValueError, "fully transparent"):
            app.crowd.load_side_walk(str(self.folder))

    def test_project_paths_legacy_and_preview_export(self):
        project = app.new_project(self.params)
        project.sequences[0].shots.append(app.Shot(self.params))
        path = self.folder.parent / "project.json"
        app.save_project_file(project, path)
        values = json.loads(path.read_text(encoding="utf-8"))
        raw = values["sequences"][0]["shots"][0]["params"]["custom_pedestrians"][0]
        self.assertEqual(raw["side_walk_folder"], self.folder.name)
        self.assertEqual(app.load_project_file(path).sequences[0].shots[0].params, self.params)
        _, preview = app._preview_images(self.params, 64, 64, 4)
        self.assertEqual(preview.tobytes(), app.ShotRenderer(self.params, supersample=2).frame(4).tobytes())
        output = list(app.sequence_outputs(project.sequences[0], self.folder.parent))
        self.assertEqual(output[4][1].tobytes(), app.ShotRenderer(self.params).frame(4).tobytes())
        raw.pop("side_walk_folder")
        raw.pop("side_walk_stride")
        path.write_text(json.dumps(values), encoding="utf-8")
        legacy = app.load_project_file(path).sequences[0].shots[0].params.custom_pedestrians[0]
        self.assertIsNone(legacy.side_walk_folder)
        self.assertEqual(legacy.side_walk_stride, app.crowd.STRIDE_FACTOR)
        raw["side_walk_folder"] = self.folder.name
        raw["side_walk_stride"] = 0
        path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stride"):
            app.load_project_file(path)
        raw["side_walk_stride"] = 1
        raw["pet_dog"] = True
        path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot use"):
            app.load_project_file(path)

    def test_cast_shadows_match_preview_and_export_with_dog(self):
        p = replace(self.params, dog_enabled=True, shadow_style="cast")
        frame = 4
        renderer = app.ShotRenderer(p)
        expected = renderer.frame(frame)
        without = app.ShotRenderer(replace(p, shadows_enabled=False)).frame(frame)
        self.assertIsNotNone(ImageChops.difference(expected, without).getbbox())
        _, preview = app._preview_images(p, p.width, p.height, frame)
        self.assertEqual(preview.tobytes(),
                         app.ShotRenderer(p, supersample=2).frame(frame).tobytes())
        sequence = app.Sequence("Cast", [app.Shot(p)])
        exported = list(app.sequence_outputs(sequence, self.folder.parent))[frame][1]
        self.assertEqual(exported.tobytes(), expected.tobytes())


class DogInteractionTests(unittest.TestCase):
    def setUp(self):
        self.spec = app.crowd.CustomPedestrian(
            "petter", ((-2, 1), (-1.5, 1)), departure=2, speed=3,
            pet_dog=True, pet_seconds=1, pet_offset=(.12, .08))
        self.p = app.Params(width=96, height=96, anim_fps=4, anim_cycles=20,
                            crowd_enabled=False, shadows_enabled=False,
                            background_perspective=False, stripe_wobble=0,
                            custom_pedestrians=(self.spec,))

    def test_phases_reverse_route_and_follow_until_both_fully_exit(self):
        for angle, tilt, size in ((0, 45, .8), (90, 60, .8), (200, 30, 3)):
            with self.subTest(angle=angle, size=size):
                p = replace(self.p, angle=angle, tilt=tilt, dog_size=size)
                renderer = app.ShotRenderer(p, supersample=1)
                interaction = renderer.interaction
                self.assertEqual(interaction.return_route, tuple(reversed(interaction.route)))
                for t in range(interaction.pet_start, interaction.return_start):
                    self.assertEqual(interaction.pose(t)[:2], interaction.stop)
                    self.assertEqual(interaction.dog_pose(t)[:2], (0, 0))
                max_step = interaction.speed / p.anim_fps + 1e-8
                for pose in (interaction.pose, interaction.dog_pose):
                    for t in range(1, interaction.end + 1):
                        self.assertLessEqual(math.dist(pose(t)[:2], pose(t - 1)[:2]), max_step)
                ped_end = interaction.return_start + interaction.duration - 1
                self.assertLess(ped_end, interaction.end)
                self.assertNotEqual(interaction.dog_pose(ped_end)[:2],
                                    interaction.dog_pose(interaction.end)[:2])
                background = Image.new("RGB", (96, 96), "#808080")
                self.assertEqual(renderer.compose(background, interaction.end).tobytes(),
                                 background.tobytes())
                walker = renderer.custom[0]
                self.assertEqual(walker.pose(interaction.end)[:2], interaction.route[0])

    def test_follow_path_visits_corners_in_order(self):
        interaction = app.interaction_for(self.p)
        corners = interaction.return_route
        self.assertEqual(interaction.dog_route, ((0, 0),) + corners)
        for index, distance in enumerate(interaction.dog_distances):
            pose = app.crowd._route_pose(interaction.dog_route, interaction.dog_distances, distance)
            np.testing.assert_allclose(pose[:2], interaction.dog_route[index], atol=1e-10)

    def test_adjusted_stop_and_output_size_independent_projection(self):
        for angle in (0, 47, 180):
            p = replace(self.p, angle=angle)
            small = app.ShotRenderer(p, 96, 95).interaction
            large = app.ShotRenderer(p, 960, 960).interaction
            np.testing.assert_allclose(small.stop, large.stop)
            np.testing.assert_allclose(small.route, large.route)
            np.testing.assert_allclose(small.stop, (
                -math.cos(math.radians(angle)) * p.person_size * .36
                - math.sin(math.radians(angle)) * p.person_size * .1 + .12,
                math.sin(math.radians(angle)) * p.person_size * .36
                - math.cos(math.radians(angle)) * p.person_size * .1 + .08))

    def test_petting_hair_hold_scale_and_strict_generic_loader(self):
        renderer = app.ShotRenderer(self.p)
        walker = renderer.custom[0]
        t = renderer.interaction.pet_start
        pose = walker.pose(t)
        sprite, anchor = walker.sprite(renderer.cam, t, *pose[2:])
        same, _ = walker.sprite(renderer.cam, t + 1, *pose[2:])
        self.assertIs(sprite, same)
        female = app.ShotRenderer(replace(self.p, custom_pedestrians=(
            replace(self.spec, female=True),)))
        other = female.custom[0].sprite(female.cam, t, *pose[2:])[0]
        self.assertEqual(sprite.tobytes(), other.tobytes())
        pet = app.crowd.load_petting()
        standing = app.crowd.load_assets()["away"]
        self.assertEqual(pet["size"][1], standing["size"][1])
        self.assertGreater(sprite.getbbox()[1], 0)
        self.assertGreater(anchor[1], sprite.getbbox()[1])
        with self.assertRaisesRegex(ValueError, "hair_female"):
            app.crowd.load_clip(str(app.crowd.ANIM_DIR / "petting"))

    def test_excited_overlay_source_registration_mirror_and_hold(self):
        assets = app.dog_mod.load_interaction_assets()
        frames, anchor = assets["excited", False]
        mirrored, mirror_anchor = assets["excited", True]
        self.assertEqual(mirror_anchor, (frames[0].width - anchor[0], anchor[1]))
        for original, other in zip(frames, mirrored):
            self.assertEqual(other.tobytes(), original.transpose(Image.FLIP_LEFT_RIGHT).tobytes())
        box, size, _ = app.dog_mod._BLINK_GEOMETRY
        scale = size[1] / (box[3] - box[1])
        with Image.open(app.crowd.ANIM_DIR / "dog_backward" / "dog_backwards0000.png") as base, \
                Image.open(app.crowd.ANIM_DIR / "dog_backwards_blink" / "dog_excited0000.png") as overlay:
            raw = Image.alpha_composite(base.convert("RGBA"), overlay.convert("RGBA"))
        raw_anchor = (box[0] + app.dog_mod.load_assets()["toward", False][1][0] / scale,
                      box[1] + app.dog_mod.load_assets()["toward", False][1][1] / scale)
        crop_x = round(raw_anchor[0] - anchor[0] / scale)
        crop_y = round(raw_anchor[1] - anchor[1] / scale)
        crop_box = (crop_x, crop_y, crop_x + round(frames[0].width / scale),
                    crop_y + round(frames[0].height / scale))
        # Compare opaque mark positions, allowing subpixel resize rounding.
        self.assertLess(raw.getbbox()[1], box[1] + app.dog_mod.PAD * 100)
        expected = raw.crop(crop_box).resize(frames[0].size, Image.LANCZOS)
        np.testing.assert_allclose(expected.getbbox(), frames[0].getbbox(), atol=1)

    def test_interaction_persistence_validation_and_legacy_defaults(self):
        raw = json.loads(json.dumps(asdict(self.p)))
        raw["custom_pedestrians"] = [asdict(self.spec)]
        decoded = app.decode_params(raw)
        self.assertEqual(decoded.custom_pedestrians, (self.spec,))
        person = raw["custom_pedestrians"][0]
        for key in ("pet_dog", "pet_seconds", "pet_offset"):
            person.pop(key)
        self.assertFalse(app.decode_params(raw).custom_pedestrians[0].pet_dog)
        for changes, text in (
                ({"dog_enabled": False}, "Show dog"),
                ({"custom_pedestrians": (self.spec, replace(self.spec, id="second"))}, "Only one"),
                ({"custom_pedestrians": (replace(self.spec, route=()),)}, "route")):
            with self.assertRaisesRegex(ValueError, text):
                app.validate_project_params(replace(self.p, **changes))
        interaction = app.interaction_for(self.p)
        conflict = app.crowd.ClipBlock("unused", interaction.pet_start, interaction.pet_start)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            app.interaction_for(replace(self.p, custom_pedestrians=(
                replace(self.spec, clips=(conflict,)),)))
        with tempfile.TemporaryDirectory() as directory:
            project = app.new_project(self.p)
            project.editor = self.p
            project.sequences[0].shots.append(app.Shot(self.p))
            path = Path(directory) / "interaction.json"
            app.save_project_file(project, path)
            loaded = app.load_project_file(path)
            self.assertEqual(loaded.editor, self.p)
            self.assertEqual(loaded.sequences[0].shots[0].params, self.p)

    def test_dynamic_crowd_clearance_random_access_and_renderer_consistency(self):
        p = replace(self.p, crowd_enabled=True, crowd_count=6)
        renderer = app.ShotRenderer(p, supersample=2)
        obstacles = app.crowd.InteractionObstacles(renderer.interaction, renderer.dog)
        for t in range(renderer.interaction.end + 1):
            for person in renderer.people:
                xy = person.ground_pose(t, obstacles)
                for obstacle in obstacles.at(t):
                    self.assertGreaterEqual(
                        math.dist(xy, (obstacle.x, obstacle.y)), obstacle.radius - 1e-8)
        frames = [0, renderer.interaction.pet_start, renderer.interaction.return_start + 5]
        expected = {t: renderer.frame(t).tobytes() for t in frames}
        for t in reversed(frames):
            self.assertEqual(renderer.frame(t).tobytes(), expected[t])
            self.assertEqual(app._preview_images(p, 96, 96, t)[1].tobytes(), expected[t])

    def test_avoidance_does_not_switch_sides_mid_encounter(self):
        p = replace(self.p, anim_fps=12, crowd_enabled=True, crowd_count=12,
                    custom_pedestrians=(replace(self.spec, speed=2, departure=0,
                                                pet_offset=(0, 0), pet_seconds=2),))
        renderer = app.ShotRenderer(p)
        obstacles = app.crowd.InteractionObstacles(renderer.interaction, renderer.dog)
        for person in renderer.people:
            poses = [person.ground_pose(t, obstacles) for t in range(renderer.interaction.end + 1)]
            backwards = [person.ground_pose(t, obstacles)
                         for t in reversed(range(renderer.interaction.end + 1))]
            self.assertEqual(poses, list(reversed(backwards)))
            for t in range(1, len(poses)):
                if math.dist(person.ground_pose(t), person.ground_pose(t - 1)) < .5:
                    self.assertLess(math.dist(poses[t], poses[t - 1]), .8)

    def test_exported_png_still_gif_and_sequence_use_same_interaction(self):
        p = replace(self.p, anim_cycles=2, custom_pedestrians=(
            replace(self.spec, walk_duration=3, pet_seconds=.5),))
        renderer = app.ShotRenderer(p)
        interaction = renderer.interaction
        self.assertLess(interaction.end, 24)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            base = str(folder / "animation")
            with patch.object(app, "_stripe_runs", return_value=[]):
                result = app._export_animation(p, base, threading.Event(), lambda _: None)
                self.assertEqual(result, "Complete")
                sequence = app.Sequence("Pet", [app.Shot(p)])
                result = app._export_sequence(sequence, folder, threading.Event(), lambda _: None)
                self.assertEqual(result, "Complete")
                for t in (interaction.pet_start, interaction.return_start + 1, interaction.end):
                    expected = app.ShotRenderer(p).frame(t)
                    for path in (folder / f"animation_{t + 1:04d}.png",
                                 folder / f"Pet_shot1.{t + 1:04d}.png"):
                        with Image.open(path) as output:
                            self.assertEqual(output.tobytes(), expected.tobytes())
                    still = folder / "still.png"
                    expected.save(still)
                    with Image.open(still) as output:
                        self.assertEqual(output.tobytes(), expected.tobytes())
            with Image.open(base + ".gif") as gif:
                self.assertNotIn("loop", gif.info)

    def test_invalid_offsets_and_missing_interaction_art_report_errors(self):
        for offset in ((float("nan"), 0), (0, float("inf"))):
            with self.assertRaisesRegex(ValueError, "petting position"):
                app.interaction_for(replace(self.p, custom_pedestrians=(
                    replace(self.spec, pet_offset=offset),)))
        app.dog_mod.load_interaction_assets.cache_clear()
        self.addCleanup(app.dog_mod.load_interaction_assets.cache_clear)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(app.crowd, "ANIM_DIR", Path(directory)):
            with self.assertRaisesRegex(ValueError, "excited dog frames"):
                app.dog_mod.load_interaction_assets()


class CrowdLoopTests(unittest.TestCase):
    def test_disabled_loop_replaces_walker_after_an_exit(self):
        params = app.Params(width=128, height=128, crowd_count=1, crowd_seed=17,
                            crowd_speed=3, anim_cycles=1)
        camera = app.Camera(params, params.width, params.height)
        person = app.crowd.make_crowd(params, app.walk_routes(params), camera)[0]
        length = (abs(person.entry) + person.route.end - person.route.start
                  + abs(person.exit))
        step = person.rate / app.crowd.FRAME_COUNT * app.crowd.STRIDE_FACTOR * person.height
        frame = math.ceil(2 * (length + 2 * step) / step)
        generation = person.appearance_generation(frame)
        self.assertGreater(generation, 0)

        original = person.frames_for_generation(0)
        replacement = person.frames_for_generation(generation)
        self.assertIsNot(original, replacement)
        self.assertNotEqual(original[0].tobytes(), replacement[0].tobytes())
        self.assertIs(replacement, person.frames_for_generation(generation))

        with patch.object(person, "frames_for_generation",
                          wraps=person.frames_for_generation) as frames_for_generation:
            app.crowd.draw_frame(Image.new("RGB", (128, 128), "white"), camera,
                                 [person], frame, loop_pedestrians=True)
            frames_for_generation.assert_not_called()
            app.crowd.draw_frame(Image.new("RGB", (128, 128), "white"), camera,
                                 [person], frame, loop_pedestrians=False)
            frames_for_generation.assert_called_once_with(generation)


class SpriteTests(unittest.TestCase):
    def test_lazy_frames_match_eager_recoloring_and_cache_only_accessed_frames(self):
        rng = np.random.default_rng(42)
        layers = {
            layer: [(rng.random((3, 4), dtype=np.float32).astype(np.float16),
                     rng.integers(0, 256, (3, 4), dtype=np.uint8))
                    for _ in range(app.crowd.FRAME_COUNT)]
            for layer in app.crowd.BODY_LAYERS + app.crowd.HAIR_LAYERS
        }
        data = {"size": (4, 3), "anchor": (1, 3), "layers": layers}
        colors = {"legs": (30, 70, 110), "arms": (170, 20, 200),
                  "head": (200, 150, 100), "hair": (40, 30, 20)}
        original_composite = Image.alpha_composite
        for female, mirror in ((False, False), (True, True)):
            with self.subTest(female=female, mirror=mirror), \
                    patch.object(app.crowd, "load_assets", return_value={"away": data}), \
                    patch.object(Image, "alpha_composite", wraps=original_composite) as composite:
                frames, anchor = app.crowd._build_frames("away", colors, female, mirror)
                composite.assert_not_called()
                first = frames[0]
                self.assertIs(frames[0], first)
                self.assertEqual(composite.call_count, 4)
                self.assertEqual(anchor, (3 if mirror else 1, 3))
                for index in range(app.crowd.FRAME_COUNT):
                    expected = Image.new("RGBA", (4, 3))
                    for layer, color in (("legs", colors["legs"]), ("arms", colors["arms"]),
                                         ("head", colors["head"]),
                                         ("hair_female" if female else "hair_male", colors["hair"])):
                        norm, alpha = layers[layer][index]
                        rgb = np.clip(norm[..., None].astype(np.float32) *
                                      np.asarray(color, dtype=np.float32), 0, 255).astype(np.uint8)
                        expected = original_composite(expected, Image.fromarray(np.dstack([rgb, alpha])))
                    if mirror:
                        expected = expected.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    self.assertEqual(frames[index].tobytes(), expected.tobytes())
                self.assertEqual(composite.call_count, 4 * app.crowd.FRAME_COUNT)


class DogTrackingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / "walk"
        self.folder.mkdir()
        Image.new("RGBA", (16, 20), "red").save(self.folder / "walk0001.png")
        self.addCleanup(app.crowd.load_side_walk.cache_clear)

    def test_tracks_custom_and_imported_walkers_in_both_directions(self):
        for folder in (None, str(self.folder)):
            with self.subTest(imported=folder is not None):
                spec = app.crowd.CustomPedestrian(
                    "walker", ((-2, 1), (2, 1)), walk_duration=12, side_walk_folder=folder)
                p = app.Params(width=96, height=96, anim_cycles=3, crowd_enabled=False,
                               dog_enabled=True, dog_blink=False, shadows_enabled=False,
                               custom_pedestrians=(spec,))
                renderer = app.ShotRenderer(p)
                gaze = renderer.dog.gaze
                for frame, sign in ((0, -1), (20, 1)):
                    with patch.object(renderer.dog, "gaze", wraps=gaze) as observed:
                        renderer.frame(frame)
                    args, kwargs = observed.call_args
                    gx, _ = gaze(*args, **kwargs)
                    self.assertGreater(sign * gx, 0.1)

    def test_tracks_approaching_petter_before_special_interaction(self):
        spec = app.crowd.CustomPedestrian(
            "petter", ((-2, 1), (-1, 1)), walk_duration=8, pet_dog=True)
        p = app.Params(width=96, height=96, anim_cycles=3, crowd_enabled=False,
                       dog_enabled=True, dog_blink=False, shadows_enabled=False,
                       custom_pedestrians=(spec,))
        renderer = app.ShotRenderer(p)
        gaze = renderer.dog.gaze
        with patch.object(renderer.dog, "gaze", wraps=gaze) as observed:
            renderer.frame(renderer.interaction.pet_start - 1)
        args, kwargs = observed.call_args
        gx, _ = gaze(*args, **kwargs)
        self.assertLess(gx, -0.1)
        with patch.object(renderer.dog, "gaze", wraps=gaze) as observed:
            renderer.frame(renderer.interaction.pet_start)
        observed.assert_not_called()

    def test_blends_custom_and_automatic_crowd_gaze(self):
        spec = app.crowd.CustomPedestrian("walker", ((-2, 1), (-1, 1)), walk_duration=12)
        p = app.Params(width=96, height=96, crowd_enabled=True, crowd_count=1,
                       dog_enabled=True, dog_blink=False, shadows_enabled=False,
                       custom_pedestrians=(spec,))
        renderer = app.ShotRenderer(p)
        walker = renderer.custom[0]
        reference = SimpleNamespace(ground_pose=lambda t, obstacle: walker.pose(t)[:2])
        gaze = renderer.dog.gaze
        expected = gaze(renderer.cam, [*renderer.people, reference], 5)
        with patch.object(renderer.dog, "gaze", wraps=gaze) as observed:
            renderer.frame(5)
        args, kwargs = observed.call_args
        np.testing.assert_allclose(gaze(*args, **kwargs), expected)

    def test_missing_import_is_not_tracked_in_editor_preview(self):
        spec = app.crowd.CustomPedestrian(
            "walker", ((-2, 1), (-1, 1)), side_walk_folder=str(self.folder / "missing"))
        p = app.Params(width=96, height=96, crowd_enabled=False, dog_enabled=True,
                       dog_blink=False, shadows_enabled=False, custom_pedestrians=(spec,))
        renderer = app.ShotRenderer(p, editor=True)
        gaze = renderer.dog.gaze
        with patch.object(renderer.dog, "gaze", wraps=gaze) as observed:
            renderer.frame(0)
        args, kwargs = observed.call_args
        self.assertEqual(gaze(*args, **kwargs), (0.0, -1.0))


class DogBlinkTests(unittest.TestCase):
    def test_blink_frames_match_normal_poses_and_anchors(self):
        normal = app.dog_mod.load_assets()
        blink = app.dog_mod.load_blink_assets()
        for mirrored in (False, True):
            normal_frames, normal_anchor = normal["toward", mirrored]
            blink_frames, blink_anchor = blink[mirrored]
            self.assertEqual(blink_anchor, normal_anchor)
            self.assertEqual(len(blink_frames), len(normal_frames))
            for index, frame in enumerate(blink_frames):
                differences = [
                    np.count_nonzero(np.any(np.asarray(frame) != np.asarray(other), axis=2))
                    for other in normal_frames
                ]
                self.assertGreater(differences[index], 0)
                self.assertEqual(index, int(np.argmin(differences)))
                if mirrored:
                    self.assertEqual(
                        frame.tobytes(),
                        blink[False][0][index].transpose(Image.FLIP_LEFT_RIGHT).tobytes())

    def test_missing_blink_assets_report_an_error(self):
        app.dog_mod.load_assets()
        app.dog_mod.load_blink_assets.cache_clear()
        self.addCleanup(app.dog_mod.load_blink_assets.cache_clear)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(app.crowd, "ANIM_DIR", Path(directory)):
            with self.assertRaisesRegex(ValueError, "dog blink frames"):
                app.dog_mod.load_blink_assets()

    def test_blinks_are_brief_irregular_and_frame_addressable(self):
        for fps in (1, 12, 24, 60):
            with self.subTest(fps=fps):
                frames = list(range(30 * fps))
                forward = [app.dog_mod.is_blinking(t, fps) for t in frames]
                backward = [app.dog_mod.is_blinking(t, fps) for t in reversed(frames)]
                self.assertEqual(forward, list(reversed(backward)))
                self.assertFalse(forward[0])
                starts = [t for t in frames if forward[t] and (t == 0 or not forward[t - 1])]
                self.assertGreaterEqual(len(starts), 8)
                self.assertGreaterEqual(starts[0], 2 * fps)
                self.assertLess(starts[0], 3 * fps)
                gaps = [b - a for a, b in zip(starts, starts[1:])]
                self.assertTrue(all(2 * fps <= gap <= 4 * fps for gap in gaps))
                if fps > 1:
                    self.assertGreater(len(set(gaps)), 1)
                duration = max(1, round(0.15 * fps))
                for start in starts:
                    self.assertTrue(all(forward[start:start + duration]))
                    if start + duration < len(forward):
                        self.assertFalse(forward[start + duration])

    def test_sprite_uses_matching_frame_direction_mirror_and_toggle(self):
        dog = app.dog_mod.Dog(height=0.8, clearance=0.9, max_tilt=12, hold=2)
        normal = app.dog_mod.load_assets()
        blink = app.dog_mod.load_blink_assets()
        with patch.object(app.dog_mod, "is_blinking", return_value=True):
            for gx in (-0.6, 0.6):
                with patch.object(dog, "gaze", return_value=(gx, -0.8)):
                    for index in range(4):
                        frame, anchor = dog.sprite(None, [], index * dog.hold)
                        expected, expected_anchor = blink[gx > 0]
                        rotated = expected[index].rotate(
                            -dog.max_tilt * gx, resample=Image.BICUBIC, center=expected_anchor)
                        self.assertEqual(anchor, expected_anchor)
                        self.assertEqual(frame.tobytes(), rotated.tobytes())
            dog.max_tilt = 0
            with patch.object(dog, "gaze", return_value=(0, 1)):
                self.assertEqual(dog.sprite(None, [], 0)[0].tobytes(),
                                 normal["away", False][0][0].tobytes())
            dog.blink_enabled = False
            with patch.object(dog, "gaze", return_value=(0, -1)), \
                    patch.object(app.dog_mod, "load_blink_assets", side_effect=AssertionError):
                self.assertEqual(dog.sprite(None, [], 0)[0].tobytes(),
                                 normal["toward", False][0][0].tobytes())

    def test_preview_still_and_exports_agree_at_blink_frame(self):
        p = app.Params(width=96, height=96, crowd_enabled=False, dog_enabled=True,
                       dog_tilt=0, anim_fps=12, anim_cycles=3)
        frame = next(t for t in range(36) if app.dog_mod.is_blinking(t, p.anim_fps))
        enabled = app.ShotRenderer(p).frame(frame)
        disabled = app.ShotRenderer(replace(p, dog_blink=False)).frame(frame)
        self.assertNotEqual(enabled.tobytes(), disabled.tobytes())
        _, preview = app._preview_images(p, p.width, p.height, frame)
        self.assertEqual(preview.tobytes(),
                         app.ShotRenderer(p, supersample=2).frame(frame).tobytes())
        background = app.render(p, frame=frame)
        app.crowd.draw_frame(background, app.Camera(p, p.width, p.height), [], frame,
                             app.make_dog(p), shadows=p.shadows_enabled,
                             shadow_style=p.shadow_style, shadow_hold=p.dog_hold)
        self.assertEqual(background.tobytes(), enabled.tobytes())
        with tempfile.TemporaryDirectory() as directory:
            sequence = app.Sequence("Blink", [app.Shot(replace(p))])
            exported = next(image for _, image, _, number in app.sequence_outputs(sequence, directory)
                            if number == frame + 1)
            self.assertEqual(exported.tobytes(), enabled.tobytes())
            base = str(Path(directory) / "animation")
            result = app._export_animation(p, base, threading.Event(), lambda progress: None)
            self.assertEqual(result, "Complete")
            with Image.open(f"{base}_{frame + 1:04d}.png") as image:
                self.assertEqual(image.tobytes(), enabled.tobytes())


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.params = app.Params(width=128, height=128, zoom=12, tilt=0)
        self.cam = app.Camera(self.params, 128, 128)
        self.sprite = Image.new("RGBA", (100, 100))
        draw = ImageDraw.Draw(self.sprite)
        draw.ellipse((35, 5, 65, 30), fill=(200, 80, 40, 220))
        draw.rectangle((30, 30, 70, 65), fill=(200, 80, 40, 180))
        draw.rectangle((30, 65, 42, 99), fill=(200, 80, 40, 220))
        draw.rectangle((58, 65, 70, 99), fill=(200, 80, 40, 220))
        self.person = SimpleNamespace(
            height=2, frames=[self.sprite] * app.crowd.FRAME_COUNT,
            anchor=(50, 100), phase=0, rate=0,
            ground_pose=lambda t, obstacle: (0, 0))
        self.dog = SimpleNamespace(
            x=0, y=0, height=2, WORK_HEIGHT=100,
            obstacle=lambda: None,
            sprite=lambda cam, people, t: (self.sprite, (50, 100)))

    def image(self, people=(), dog=None, **options):
        image = Image.new("RGB", (128, 128), "white")
        return app.crowd.draw_frame(image, self.cam, people, 0, dog, **options)

    def test_both_characters_have_shadows_and_disabled_is_unchanged(self):
        for people, dog in (([self.person], None), ([], self.dog)):
            with self.subTest(dog=dog is not None):
                self.cam = app.Camera(replace(self.params, tilt=45), 128, 128)
                disabled = self.image(people, dog)
                self.assertEqual(disabled.tobytes(),
                                 self.image(people, dog, shadows=False).tobytes())
                for style in ("contact", "cast"):
                    enabled = self.image(people, dog, shadows=True, shadow_style=style)
                    self.assertIsNotNone(ImageChops.difference(disabled, enabled).getbbox())

    def test_shadow_follows_sprite_alpha_and_animation(self):
        blank = Image.new("RGBA", self.sprite.size)
        self.person.frames = [blank] * app.crowd.FRAME_COUNT
        empty = self.image([self.person], shadows=True, shadow_style="cast")
        self.assertIsNone(ImageChops.difference(
            empty, Image.new("RGB", empty.size, "white")).getbbox())
        self.person.frames[1] = self.sprite
        self.person.rate = 1
        image = Image.new("RGB", (128, 128), "white")
        animated = app.crowd.draw_frame(image, self.cam, [self.person], 1,
                                        shadows=True, shadow_style="cast")
        without = Image.new("RGB", (128, 128), "white")
        app.crowd.draw_frame(without, self.cam, [self.person], 1)
        self.assertIsNotNone(ImageChops.difference(animated, without).getbbox())

    def test_contact_shadow_is_a_blob_independent_of_sprite_shape(self):
        cam = app.Camera(self.params, 512, 512)
        other = Image.new("RGBA", (100, 100))
        ImageDraw.Draw(other).rectangle((5, 5, 20, 50), fill=(0, 0, 0, 255))
        masks = []
        for sprite in (self.sprite, other):
            mask = Image.new("L", (512, 512))
            app.crowd._draw_shadow(mask, cam, sprite, (50, 100), 100,
                                   0, 0, 2, 0, "contact", "test", 0, 100)
            masks.append(mask)
        self.assertEqual(masks[0].tobytes(), masks[1].tobytes())
        bounds = masks[0].getbbox()
        self.assertIsNotNone(bounds)
        self.assertGreater(masks[0].getpixel((256, 256)), 0)
        filled = sum(value > 0 for value in masks[0].crop(bounds).getdata())
        self.assertGreater(filled, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 0.45)
        empty = Image.new("L", (512, 512))
        app.crowd._draw_shadow(empty, cam, Image.new("RGBA", (100, 100)),
                               (50, 100), 100, 0, 0, 2, 0, "contact", "test", 0, 100)
        self.assertIsNone(empty.getbbox())

    def test_cast_shadow_projects_head_away_from_feet_and_preserves_alpha(self):
        def shadow(box, alpha):
            sprite = Image.new("RGBA", (100, 100))
            ImageDraw.Draw(sprite).rectangle(box, fill=(0, 0, 0, alpha))
            mask = Image.new("L", (128, 128))
            app.crowd._draw_shadow(mask, self.cam, sprite, (50, 100), 100,
                                   0, 0, 2, 0, "cast", "test", 0,
                                   app.crowd._walker_projection(self.cam, 0, 0, 2)[3])
            return mask
        head = shadow((45, 5, 55, 15), 255)
        feet = shadow((45, 85, 55, 95), 255)
        head_box, feet_box = head.getbbox(), feet.getbbox()
        self.assertIsNotNone(head_box)
        self.assertIsNotNone(feet_box)
        self.assertGreater(head_box[0], feet_box[0] + 5)
        self.assertLess(head_box[1], feet_box[1] - 5)
        lighter = shadow((45, 5, 55, 15), 64)
        self.assertLess(lighter.getextrema()[1], head.getextrema()[1] / 2)

    def test_cast_shadow_meets_both_feet_at_camera_angles(self):
        sprite = Image.new("RGBA", (100, 100))
        draw = ImageDraw.Draw(sprite)
        for x in (20, 80):
            draw.rectangle((x - 5, 92, x + 5, 99), fill=(200, 80, 40, 255))
        for tilt in (20, 45, 70):
            for angle in (0, 45, 90, 135, 180, 270):
                with self.subTest(tilt=tilt, angle=angle):
                    cam = app.Camera(replace(self.params, tilt=tilt, angle=angle), 512, 512)
                    for pixel_height in (
                            app.crowd._walker_projection(cam, 0, 0, 2)[3],
                            app.crowd._project(sprite, cam, 0, 0, 2)[3]):
                        mask = Image.new("L", (512, 512))
                        app.crowd._draw_shadow(mask, cam, sprite, (50, 100), 100,
                                               0, 0, 2, 0, "cast", "test", 0, pixel_height)
                        fx, fy = cam.project(cam.to_cam(0, 0))
                        for x in (20, 80):
                            px = round(fx + (x - 50) * pixel_height / 100)
                            py = round(fy - 2 * pixel_height / 100)
                            near = mask.crop((px - 4, py - 4, px + 5, py + 5))
                            self.assertGreater(near.getextrema()[1], 0)

    def test_shadow_preserves_sprite_left_right_orientation(self):
        for angle in (0, 45, 90, 180, 270):
            cam = app.Camera(replace(self.params, angle=angle, tilt=45), 512, 512)
            along = 2 * 0.85 * (100 - 45) / 100 / 2**0.5
            center_x = cam.project(cam.to_cam(along, along))[0]
            for box, sign in (((70, 40, 90, 50), 1), ((10, 40, 30, 50), -1)):
                with self.subTest(angle=angle, sign=sign):
                    sprite = Image.new("RGBA", (100, 100))
                    ImageDraw.Draw(sprite).rectangle(box, fill=(0, 0, 0, 255))
                    mask = Image.new("L", (512, 512))
                    app.crowd._draw_shadow(mask, cam, sprite, (50, 100), 100,
                                           0, 0, 2, 0, "cast", "test", 0,
                                           app.crowd._walker_projection(cam, 0, 0, 2)[3])
                    bounds = mask.getbbox()
                    self.assertIsNotNone(bounds)
                    self.assertGreater(sign * ((bounds[0] + bounds[2]) / 2 - center_x), 1)

    def test_styles_and_drawing_hold_are_deterministic(self):
        def frame(number, style):
            image = Image.new("RGB", (128, 128), "white")
            return app.crowd.draw_frame(image, self.cam, [self.person], number,
                                        shadows=True, shadow_style=style, shadow_hold=2)
        contact = frame(0, "contact")
        self.assertEqual(contact.tobytes(), frame(1, "contact").tobytes())
        self.assertEqual(contact.tobytes(), frame(0, "contact").tobytes())
        self.assertNotEqual(contact.tobytes(), frame(2, "contact").tobytes())
        self.assertNotEqual(contact.tobytes(), frame(0, "cast").tobytes())

    def test_cast_shadow_visible_when_sprite_is_offscreen(self):
        self.person.anchor = (100, 100)
        self.person.ground_pose = lambda t, obstacle: (-6.5, 0)
        without = self.image([self.person])
        with_shadow = self.image([self.person], shadows=True, shadow_style="cast")
        self.assertIsNotNone(ImageChops.difference(without, with_shadow).getbbox())

    def test_extreme_camera_and_near_plane(self):
        for tilt, x, y in ((0, 0, 0), (70, 0, 0), (70, 0, -30)):
            with self.subTest(tilt=tilt, x=x, y=y):
                self.cam = app.Camera(replace(self.params, tilt=tilt), 128, 128)
                self.person.ground_pose = lambda t, obstacle: (x, y)
                image = self.image([self.person], shadows=True, shadow_style="cast")
                self.assertEqual(image.size, (128, 128))


class SequenceUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.paths = []
        for color in ("red", "blue"):
            path = self.directory / f"{color}.png"
            Image.new("RGB", (20, 20), color).save(path)
            self.paths.append(str(path))
        self.params = app.Params(width=16, height=16, crowd_enabled=False,
                                 dog_enabled=False, stripe_wobble=0, anim_cycles=1,
                                 background_perspective=False,
                                 background_images=tuple(self.paths), dog_hold=2)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.no_stripes = patch.object(app, "_stripe_runs", return_value=[])
        self.no_stripes.start()
        self.addCleanup(self.no_stripes.stop)
        with patch.object(app, "load_defaults", return_value=replace(self.params)), \
                patch.object(app, "_BackgroundWorker", _ThreadWorker):
            self.ui = app.App(self.root)
        self.addCleanup(self._cancel_jobs)
        self.addCleanup(app._road_texture.cache_clear)
        self.addCleanup(app._tiled_road_texture.cache_clear)
        self.addCleanup(self._shutdown_worker)
        self._cancel_jobs()

    def _shutdown_worker(self):
        self.ui._closed = True
        if self.ui._worker_cancel is not None:
            self.ui._worker_cancel.set()
        self.ui._executor.shutdown(wait=True, cancel_futures=True)

    def test_cycles_spinner_supports_500_cycle_shots(self):
        widgets = [self.root]
        spinbox = None
        while widgets:
            widget = widgets.pop()
            widgets.extend(widget.winfo_children())
            if (widget.winfo_class() == "TSpinbox"
                    and str(widget.cget("textvariable")) == str(self.ui.vars["anim_cycles"])):
                spinbox = widget
                break
        self.assertIsNotNone(spinbox)
        self.assertEqual(int(float(spinbox.cget("to"))), 500)
        self.ui.vars["anim_cycles"].set(500)
        self.ui.add_shot()
        shot = self.ui.sequence.shots[0]
        self.assertEqual(shot.params.anim_cycles, 500)
        path = self.directory / "long-shot.json"
        app.save_project_file(self.ui.project, path)
        loaded = app.load_project_file(path)
        self.assertEqual(loaded.sequences[0].shots[0].params.anim_cycles, 500)
        self.assertEqual(shot.params.anim_cycles * app.crowd.FRAME_COUNT, 6000)

    def test_imported_pedestrian_controls_and_relink(self):
        ui = self.ui
        folder = self.directory / "walker"
        folder.mkdir()
        Image.new("RGBA", (8, 8), "red").save(folder / "walk0001.png")
        with patch.object(app.filedialog, "askdirectory", return_value=str(folder)):
            ui._add_side_pedestrian()
        self.assertEqual(ui._pedestrian().side_walk_folder, str(folder))
        self.assertTrue(all(str(button.cget("state")) == "disabled"
                            for button in (*ui.builtin_look_buttons, *ui.clip_buttons, *ui.pet_controls)))
        self.assertTrue(all(str(button.cget("state")) == "normal" for button in ui.side_buttons))
        with patch.object(app.simpledialog, "askfloat", return_value=1.5):
            ui._set_side_stride()
        self.assertEqual(ui._pedestrian().side_walk_stride, 1.5)
        other = self.directory / "replacement"
        other.mkdir()
        Image.new("RGBA", (8, 8), "blue").save(other / "walk0001.png")
        ui._change_pedestrian(side_walk_folder=str(self.directory / "missing"))
        self.assertIn("[missing: relink]", ui.pedestrian_hint.get())
        with patch.object(app.filedialog, "askdirectory", return_value=str(other)):
            ui._relink_side_walk()
        self.assertEqual(ui._pedestrian().side_walk_folder, str(other))
        ui._add_pedestrian()
        self.assertTrue(all(str(button.cget("state")) == "normal"
                            for button in (*ui.builtin_look_buttons, *ui.clip_buttons, *ui.pet_controls)))
        self.assertTrue(all(str(button.cget("state")) == "disabled" for button in ui.side_buttons))

    def test_imported_placement_and_switch_to_route(self):
        ui = self.ui
        ui.notebook.select(ui.pedestrian_tab)
        folder = self.directory / "stationary"
        folder.mkdir()
        Image.new("RGBA", (8, 8), "red").save(folder / "frame0001.png")
        with patch.object(app.filedialog, "askdirectory", return_value=str(folder)):
            ui._add_side_pedestrian()
        ui._preview_cam = app.Camera(ui.p, 160, 160)
        ui._preview_bounds = (10, 20, 170, 180)
        ui._start_route()
        ui._place_route_point(SimpleNamespace(x=80, y=100))
        self.assertIn("animate in place", ui.pedestrian_hint.get())
        ui._finish_route()
        self.assertEqual(len(ui._pedestrian().route), 1)
        self.assertEqual(list(ui._timeline_blocks(ui._pedestrian())), [])
        self.assertEqual(str(ui.speed_button.cget("state")), "disabled")
        self.assertEqual(str(ui.side_buttons[1].cget("state")), "disabled")
        saved = ui._pedestrian()
        ui.add_shot()
        ui._start_route()
        ui._place_route_point(SimpleNamespace(x=70, y=100))
        ui._place_route_point(SimpleNamespace(x=110, y=105))
        ui._finish_route()
        self.assertEqual(len(ui._pedestrian().route), 2)
        self.assertGreater(ui._pedestrian().walk_duration, 1)
        self.assertEqual(str(ui.speed_button.cget("state")), "normal")
        ui._start_route()
        ui._place_route_point(SimpleNamespace(x=100, y=105))
        ui._finish_route()
        self.assertEqual(len(ui._pedestrian().route), 1)
        self.assertIsNone(ui._pedestrian().walk_duration)
        self.assertNotEqual(ui._pedestrian().route, saved.route)
        ui.edit_shot(ui.sequence.shots[0].id)
        self.assertEqual(ui.p.custom_pedestrians, (saved,))

    def test_camera_shift_reframes_without_tilting_and_restores_with_shot(self):
        ui = self.ui
        ui.vars["camera_shift"].set(20)
        ui.add_shot()
        ui.vars["camera_shift"].set(-20)
        ui.edit_shot(ui.sequence.shots[0].id)
        params = ui.read_params()
        self.assertEqual(params.camera_shift, 20)
        for tilt in (0, 70):
            with self.subTest(tilt=tilt):
                point = (0, 0)
                base = app.Camera(replace(params, tilt=tilt, camera_shift=0), 200, 150)
                for shift in (-20, 20):
                    cam = app.Camera(replace(params, tilt=tilt, camera_shift=shift), 200, 150)
                    self.assertEqual((cam.fwd, cam.up), (base.fwd, base.up))
                    x, y = cam.project(cam.to_cam(*point))
                    np.testing.assert_allclose((x, y), (100, 75 + shift * 2))
                    np.testing.assert_allclose(cam.ground_point(x, y), point, atol=1e-8)

    def test_pet_position_drag_nudge_reset_fit_and_shot_edit(self):
        ui = self.ui
        ui.notebook.select(ui.pedestrian_tab)
        ui.vars["dog_enabled"].set(True)
        ui._add_pedestrian()
        ui._change_pedestrian(route=((-2, 1), (-1.5, 1)), pet_dog=True, pet_seconds=1)
        ui.vars["angle"].set(47)
        ui.vars["tilt"].set(60)
        ui.read_params()
        ui._preview_cam = app.Camera(ui.p, 160, 160)
        ui._preview_bounds = (10, 20, 170, 180)
        ui._fit_interaction()
        self.assertGreaterEqual(ui._timeline_frames(), ui._selected_interaction().end + 1)
        ui._position_pet()
        self.assertTrue(ui._positioning_pet)
        self.assertEqual(ui._scrub_frame, ui._selected_interaction().pet_start)
        cam = ui._preview_cam
        original = cam.project(cam.to_cam(*ui._selected_interaction().stop))
        ui.pet_step_var.set("2")
        ui._nudge_pet(1, -1)
        moved = cam.project(cam.to_cam(*ui._selected_interaction().stop))
        np.testing.assert_allclose(moved, (original[0] + 2, original[1] - 2), atol=1e-8)
        ui._place_route_point(SimpleNamespace(x=100, y=100))
        ui._drag_pet_position(SimpleNamespace(x=103, y=104))
        dragged = cam.project(cam.to_cam(*ui._selected_interaction().stop))
        np.testing.assert_allclose(dragged, (moved[0] + 3, moved[1] + 4), atol=1e-8)
        route = ui._pedestrian().route
        ui._reset_pet_position()
        self.assertEqual(ui._pedestrian().pet_offset, (0, 0))
        self.assertEqual(ui._pedestrian().route, route)
        ui._nudge_pet(-1, 0)
        saved = ui._pedestrian()
        ui.add_shot()
        ui._change_pedestrian(pet_offset=(0, 0))
        ui.edit_shot(ui.sequence.shots[0].id)
        self.assertEqual(ui.p.custom_pedestrians[0], saved)

    def test_pet_position_closeup_hides_routes_and_preserves_drag_scale(self):
        ui = self.ui
        ui.notebook.select(ui.pedestrian_tab)
        ui.vars["dog_enabled"].set(True)
        ui._add_pedestrian()
        ui._change_pedestrian(route=((-2, 1), (-1, 1)), pet_dog=True)
        ui._fit_interaction()
        ui._position_pet()
        params = replace(ui.read_params())
        frame = ui._scrub_frame
        result = app._pet_preview_images(params, 160, 160, frame)
        self.assertGreater(result[2], 1)
        self.assertEqual(result[1].size, (160, 160))
        normal = app._preview_images(params, 160, 160, frame)[1]
        self.assertNotEqual(result[1].tobytes(), normal.tobytes())
        context = (ui._preview_revision, params, 160, 160, frame, 180, 180)
        ui._work_finished("preview", context, result, False)
        self.assertEqual(ui.canvas.find_withtag("route"), ())
        self.assertEqual(ui.read_params(), params)
        cam = ui._preview_cam
        before = cam.project(cam.to_cam(*ui._selected_interaction().stop))
        ui._move_pet_pixels(4, -4)
        after = cam.project(cam.to_cam(*ui._selected_interaction().stop))
        np.testing.assert_allclose(
            after, (before[0] + 4 / result[2], before[1] - 4 / result[2]), atol=1e-8)
        ui._position_pet()
        self.assertFalse(ui._positioning_pet)
        self.assertIsNotNone(ui._job)
        params = replace(ui.read_params())
        normal = app._preview_images(params, 160, 160, ui._scrub_frame)
        context = (ui._preview_revision, params, 160, 160, ui._scrub_frame, 180, 180)
        ui._work_finished("preview", context, normal, False)
        self.assertEqual(ui._preview_scale, 1)
        self.assertTrue(ui.canvas.find_withtag("route"))

    def test_custom_pedestrian_route_timeline_and_shot_edit(self):
        ui = self.ui
        ui.notebook.select(ui.pedestrian_tab)
        self.assertEqual(ui.pedestrian_list.size(), 0)
        ui._add_pedestrian()
        ui._start_route()
        ui._preview_cam = app.Camera(ui.p, 160, 160)
        ui._preview_bounds = (10, 20, 170, 180)
        for x, y in ((2, 2), (70, 100), (90, 110), (100, 105)):
            ui._place_route_point(SimpleNamespace(x=x, y=y))
        ui._undo_route_point()
        ui._place_route_point(SimpleNamespace(x=110, y=105))
        ui._finish_route()
        self.assertEqual(len(ui._pedestrian().route), 3)
        with patch.object(app.colorchooser, "askcolor", return_value=((255, 0, 0), "#ff0000")), \
                patch.object(app.simpledialog, "askfloat", side_effect=(2.5, 1.25)):
            ui._pick_part(0)
            ui._set_pedestrian_speed()
            ui._set_pedestrian_size()
        with patch.object(ui.timeline, "winfo_width", return_value=311):
            ui._timeline_press(SimpleNamespace(x=133, y=12))
            ui._timeline_release(SimpleNamespace(x=133, y=12))
        folder = str(app.crowd.ANIM_DIR / "walk_forward")

        def submit_clip(dialog, _window):
            dialog.folder.set(folder)
            dialog.start.set("6")
            dialog.end.set("8")
            dialog.ok()

        with patch.object(app.ClipDialog, "wait_window", submit_clip):
            ui._add_clip()
        ui.add_shot()
        shot = ui.sequence.shots[0]
        spec = shot.params.custom_pedestrians[0]
        self.assertEqual((ui._scrub_frame, spec.colors[0], spec.speed), (3, "#ff0000", 2.5))
        self.assertEqual(spec.height, 1.25)
        self.assertEqual(spec.clips[0].start, 5)
        path = self.directory / "custom.json"
        app.save_project_file(ui.project, path)
        ui._install_project(app.load_project_file(path), path)
        ui.edit_shot(shot.id)
        self.assertEqual(ui.p.custom_pedestrians, (spec,))

    def test_timeline_drags_walk_and_overlapping_animation_lanes(self):
        ui = self.ui
        ui._add_pedestrian()
        spec = app.crowd
        ui._change_pedestrian(route=((0, 0), (1, 0)), departure=1, walk_duration=5,
                              clips=(spec.ClipBlock("gesture", 0, 1),
                                     spec.ClipBlock("wave", 9, 10)))

        def drag(boundary, y, frames):
            x = round(ui._timeline_x(boundary))
            ui._timeline_press(SimpleNamespace(x=x, y=y))
            end = SimpleNamespace(x=x + 20 * frames, y=y)
            ui._timeline_drag(end)
            ui._timeline_release(end)

        with patch.object(ui.timeline, "winfo_width", return_value=311):
            drag(3.5, 34, 2)  # Move Walk from frames 1–5 to 3–7 (zero-based).
            self.assertEqual((ui._pedestrian().departure, ui._pedestrian().walk_duration), (3, 5))
            speed = ui._pedestrian().walk_speed(ui.p.anim_fps)
            drag(7.85, 34, -2)
            self.assertEqual(ui._pedestrian().walk_duration, 3)
            self.assertAlmostEqual(ui._pedestrian().walk_speed(ui.p.anim_fps), 2 * speed)
            drag(3.1, 34, -1)
            self.assertEqual((ui._pedestrian().departure, ui._pedestrian().walk_duration), (2, 4))
            self.assertLess(ui._pedestrian().walk_speed(ui.p.anim_fps), 2 * speed)
            drag(5.85, 34, -3)  # One-frame walks cannot show both endpoints.
            self.assertEqual(ui._pedestrian().walk_duration, 4)
            self.assertIn("at least 2", ui.timeline_help.cget("text"))
            drag(4, 34, -3)
            self.assertEqual(ui._pedestrian().departure, 2)
            self.assertIn("inside the shot", ui.timeline_help.cget("text"))
            drag(.8, 60, 3)  # Animation overlaps Walk but is on its own lane.
            drag(4.85, 60, 2)
            clip = ui._pedestrian().clips[0]
            self.assertEqual((clip.start, clip.end), (3, 6))
            drag(6.85, 60, 3)  # Would touch the next animation.
            self.assertEqual(ui._pedestrian().clips[0], clip)
            self.assertIn("cannot overlap", ui.timeline_help.cget("text"))

    def _cancel_jobs(self):
        for name in ("_job", "_anim_job", "_worker_job", "_cards_job"):
            job = getattr(self.ui, name)
            if job is not None:
                self.root.after_cancel(job)
                setattr(self.ui, name, None)

    def _start_render(self):
        with patch.object(app.filedialog, "askdirectory", return_value=str(self.directory)):
            self.ui.render_sequence()

    def _drain_render(self):
        self._wait_until(lambda: not self.ui._rendering)

    def _wait_until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "Background operation timed out")
            self.root.update()
            time.sleep(0.002)

    def _assert_heartbeat_until_idle(self, release):
        beats = []
        job = None

        def beat():
            nonlocal job
            beats.append(time.monotonic())
            job = self.root.after(10, beat)

        beat()
        release_job = self.root.after(200, release.set)
        try:
            self._drain_render()
        finally:
            release.set()
            self.root.after_cancel(release_job)
            self.root.after_cancel(job)
        self.assertGreaterEqual(len(beats), 8)
        self.assertLess(max(b - a for a, b in zip(beats, beats[1:])), 0.250)

    def test_add_snapshots_frame_zero_and_explicit_update_after_reorder(self):
        self.ui.vars["animate_preview"].set(True)
        self.ui._anim_t = 3
        self.ui.add_shot()
        first = self.ui.sequence.shots[0]
        self._wait_until(lambda: first.id in self.ui._thumbnails)
        thumbnail = ImageTk.getimage(self.ui._thumbnails[first.id])
        self.assertEqual(thumbnail.getpixel((40, 40))[:3], (255, 0, 0))
        self.ui.vars["angle"].set(45)
        self.ui.add_shot()
        self.assertEqual(first.params.angle, 0)
        self.assertIsNot(first.params, self.ui.p)
        self.ui.edit_shot(first.id)
        self.assertFalse(self.ui.vars["animate_preview"].get())
        self.assertEqual(self.ui._preview_frame, 0)
        self.ui.vars["angle"].set(90)
        self.assertEqual(first.params.angle, 0)
        self.ui.move_shot(first.id, 1)
        self.assertEqual(self.ui.sequence.shots[1].id, first.id)
        self.assertTrue(self.ui.update_shot())
        self.assertEqual(first.params.angle, 90)
        self.assertEqual(self.ui.sequence.shots[0].params.angle, 45)

    def test_drag_reorders_cards_and_removal_clears_edit_target(self):
        for angle in (0, 45, 90):
            self.ui.vars["angle"].set(angle)
            self.ui.add_shot()
        first = self.ui.sequence.shots[0]
        self.ui.edit_shot(first.id)
        self.root.update_idletasks()
        self.ui._start_drag(first.id)
        last_card = self.ui._card_widgets[-1]
        self.ui._drag_motion(SimpleNamespace(
            x_root=last_card.winfo_rootx() + last_card.winfo_width()))
        self.assertEqual(self.ui._drag_target, 3)
        self.ui._drop_shot(SimpleNamespace())
        self.assertEqual([s.params.angle for s in self.ui.sequence.shots], [45, 90, 0])
        self.assertEqual(self.ui._edit_id, first.id)
        with patch.object(app.messagebox, "askyesno", return_value=True):
            self.ui.remove_shot(first.id)
        self.assertIsNone(self.ui._edit_id)
        self.assertNotIn(first.id, self.ui._thumbnails)

    def test_pending_shot_edits_can_cancel_switch_or_be_discarded(self):
        self.ui.add_shot()
        self.ui.add_shot()
        first, second = self.ui.sequence.shots
        self.ui.edit_shot(first.id)
        self.ui.vars["angle"].set(60)
        with patch.object(app.messagebox, "askyesnocancel", return_value=None):
            self.ui.edit_shot(second.id)
        self.assertEqual(self.ui._edit_id, first.id)
        self.assertEqual(self.ui.p.angle, 60)
        with patch.object(app.messagebox, "askyesnocancel", return_value=False):
            self.ui.edit_shot(second.id)
        self.assertEqual(self.ui._edit_id, second.id)
        self.assertEqual(first.params.angle, 0)
        self.assertEqual(self.ui.p.angle, 0)

    def test_save_keeps_uncommitted_editor_separate_and_open_restores(self):
        self.ui.add_shot()
        shot = self.ui.sequence.shots[0]
        self.ui.edit_shot(shot.id)
        self.ui.vars["angle"].set(60)
        path = self.directory / "saved.json"
        with patch.object(app.filedialog, "asksaveasfilename", return_value=str(path)):
            self.assertTrue(self.ui.save_project())
        loaded = app.load_project_file(path)
        self.assertEqual(loaded.editor.angle, 60)
        self.assertEqual(loaded.sequences[0].shots[0].params.angle, 0)
        self.assertTrue(self.ui._allow_replace_project())
        with patch.object(app.filedialog, "askopenfilename", return_value=str(path)):
            self.ui.open_project()
            self._drain_render()
        self.assertEqual(self.ui.p.angle, 60)
        self.assertIsNone(self.ui._edit_id)
        self.assertEqual(self.ui.sequence.shots[0].params.angle, 0)

    def test_failed_open_and_cancelled_new_preserve_project(self):
        self.ui.add_shot()
        project = self.ui.project
        invalid = self.directory / "bad.json"
        invalid.write_text("{}")
        with patch.object(app.filedialog, "askopenfilename", return_value=str(invalid)), \
                patch.object(app.messagebox, "showerror") as error:
            self.ui.open_project()
            self._drain_render()
        error.assert_called_once()
        self.assertIs(self.ui.project, project)
        with patch.object(app.messagebox, "askyesnocancel", return_value=None):
            self.ui.new_project()
        self.assertIs(self.ui.project, project)

    def test_named_sequences_reject_duplicates_and_save_active_selection(self):
        with patch.object(app.simpledialog, "askstring", return_value="Other"):
            self.ui.add_sequence()
        self.ui.add_shot()
        other = self.ui.sequence
        with patch.object(app.simpledialog, "askstring", return_value="sequence1"), \
                patch.object(app.messagebox, "showerror") as error:
            self.ui.rename_sequence()
        error.assert_called_once()
        self.assertEqual(other.name, "Other")
        self.ui.sequence_choice.current(0)
        self.ui._select_sequence()
        self.assertEqual(self.ui.sequence.name, "Sequence1")
        self.assertEqual(other.shots[0].params, self.ui.p)
        self.assertEqual(app.project_values(self.ui.project, self.directory)["active_sequence"],
                         self.ui.sequence.id)

    def test_export_exact_names_order_sizes_and_frame_reset(self):
        self.ui.add_shot()
        self.ui.p.background_images = (self.paths[1], self.paths[0])
        self.ui.vars["width"].set(32)
        self.ui.vars["anim_cycles"].set(2)
        self.ui.vars["anim_fps"].set(24)
        self.ui.add_shot()
        self.ui.move_shot(self.ui.sequence.shots[1].id, -1)
        self._start_render()
        self.assertTrue(self.ui._rendering)
        self._drain_render()
        expected = {f"Sequence1_shot{shot}.{frame:04d}.png"
                    for shot, total in ((1, 24), (2, 12)) for frame in range(1, total + 1)}
        self.assertEqual({p.name for p in self.directory.glob("Sequence*.png")}, expected)
        for filename, size, color in (
                ("Sequence1_shot1.0001.png", (32, 16), (0, 0, 255)),
                ("Sequence1_shot1.0003.png", (32, 16), (255, 0, 0)),
                ("Sequence1_shot2.0001.png", (16, 16), (255, 0, 0))):
            with Image.open(self.directory / filename) as image:
                self.assertEqual(image.size, size)
                self.assertEqual(image.getpixel((8, 8)), color)
        self.assertIn("Complete: 36/36", self.ui.status.cget("text"))
        self.assertEqual(str(self.ui.sequence_choice.cget("state")), "readonly")

    def test_static_shot_and_conflicting_destination(self):
        self.ui.p.background_images = (self.paths[0],)
        self.ui.add_shot()
        outputs = list(app.sequence_outputs(self.ui.sequence, self.directory))
        self.assertEqual(len(outputs), 12)
        self.assertEqual(outputs[0][1].tobytes(), outputs[-1][1].tobytes())
        existing = self.directory / "Sequence1_shot99.9999.png"
        existing.write_bytes(b"do not replace")
        with patch.object(app.messagebox, "showerror") as error:
            self._start_render()
            self._drain_render()
        error.assert_called_once()
        self.assertFalse(self.ui._rendering)
        self.assertEqual(existing.read_bytes(), b"do not replace")
        self.assertEqual(list(self.directory.glob("Sequence1_shot1.*.png")), [])

    def test_shot_renderer_matches_existing_crowd_and_dog_composition(self):
        self.no_stripes.stop()
        p = replace(self.params, width=64, height=64, crowd_enabled=True,
                    crowd_count=2, dog_enabled=True, stripe_wobble=0.25)
        renderer = app.ShotRenderer(p)
        self.assertEqual(len(renderer.people), 2)
        for style, number in (("contact", 0), ("contact", 5), ("cast", 5)):
            with self.subTest(style=style, frame=number):
                p.shadow_style = style
                renderer = app.ShotRenderer(p)
                bg = app.render(p, frame=number)
                expected = self.ui._compose_crowd(
                    bg, p, app.Camera(p, p.width, p.height), number)
                actual = renderer.frame(number)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertNotEqual(actual.tobytes(), bg.tobytes())
                if number == 0:
                    sequence = app.Sequence("Test", [app.Shot(replace(p))])
                    exported = next(app.sequence_outputs(sequence, self.directory))[1]
                    self.assertEqual(exported.tobytes(), actual.tobytes())

    def test_shadow_controls_follow_shot_edits(self):
        self.ui.vars["shadows_enabled"].set(False)
        self.ui.vars["shadow_style"].set("cast")
        self.ui.add_shot()
        shot = self.ui.sequence.shots[0]
        self.assertFalse(shot.params.shadows_enabled)
        self.assertEqual(shot.params.shadow_style, "cast")
        self.ui.vars["shadows_enabled"].set(True)
        self.ui.vars["shadow_style"].set("contact")
        self.ui.edit_shot(shot.id)
        self.assertFalse(self.ui.vars["shadows_enabled"].get())
        self.assertEqual(self.ui.vars["shadow_style"].get(), "cast")

    def test_blink_toggle_follows_shot_edits(self):
        self.assertTrue(self.ui.vars["dog_blink"].get())
        widgets = [self.root]
        checkbox = None
        while widgets:
            widget = widgets.pop()
            widgets.extend(widget.winfo_children())
            if (widget.winfo_class() == "TCheckbutton"
                    and widget.cget("text") == "Dog blinks occasionally"):
                checkbox = widget
                break
        self.assertIsNotNone(checkbox)
        checkbox.invoke()
        self.assertFalse(self.ui.vars["dog_blink"].get())
        self.assertIsNotNone(self.ui._job)
        self.ui.add_shot()
        shot = self.ui.sequence.shots[0]
        self.assertFalse(shot.params.dog_blink)
        self.ui.vars["dog_blink"].set(True)
        self.ui.edit_shot(shot.id)
        self.assertFalse(self.ui.vars["dog_blink"].get())
        self.assertFalse(self.ui.read_params().dog_blink)

    def test_background_toggle_follows_shot_edits(self):
        widgets = [self.root]
        checkboxes = {}
        while widgets:
            widget = widgets.pop()
            widgets.extend(widget.winfo_children())
            if (widget.winfo_class() == "TCheckbutton"
                    and widget.cget("text") in ("Animate background", "Animate stripes")):
                checkboxes[widget.cget("text")] = widget
        self.assertEqual(len(checkboxes), 2)
        checkboxes["Animate background"].invoke()
        self.assertIsNotNone(self.ui._job)
        self.ui.add_shot()
        shot = self.ui.sequence.shots[0]
        self.assertFalse(shot.params.background_animated)
        self.assertTrue(shot.params.stripes_animated)
        self.ui.vars["background_animated"].set(True)
        checkboxes["Animate stripes"].invoke()
        self.assertFalse(self.ui.vars["stripes_animated"].get())
        self.ui.edit_shot(shot.id)
        self.assertFalse(self.ui.vars["background_animated"].get())
        self.assertFalse(self.ui.read_params().background_animated)
        self.assertTrue(self.ui.vars["stripes_animated"].get())
        self.assertTrue(self.ui.read_params().stripes_animated)

    def test_cancellation_retains_frames_and_restores_controls(self):
        self.ui.add_shot()
        saved = threading.Event()
        release = threading.Event()
        original_save = Image.Image.save

        def delayed_save(image, *args, **kwargs):
            original_save(image, *args, **kwargs)
            saved.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the PNG write")

        with patch.object(Image.Image, "save", delayed_save):
            try:
                self._start_render()
                self._wait_until(saved.is_set)
                self.ui.cancel_sequence_render()
                self.assertTrue(self.ui._rendering)
                self.assertEqual(str(self.ui.sequence_choice.cget("state")), "disabled")
            finally:
                release.set()
            self._drain_render()
        self.assertFalse(self.ui._rendering)
        self.assertEqual(len(list(self.directory.glob("Sequence*.png"))), 1)
        self.assertIn("Cancelled: 1/12", self.ui.status.cget("text"))
        self.assertEqual(str(self.ui.cancel_render_button.cget("state")), "disabled")

    def test_scheduled_render_completes_through_tk_event_loop(self):
        self.ui.add_shot()
        self._start_render()

        def finish_when_idle():
            if self.ui._rendering:
                self.root.after(5, finish_when_idle)
            else:
                self.root.quit()

        watchdog = self.root.after(5000, self.root.quit)
        self.root.after(5, finish_when_idle)
        self.root.mainloop()
        self.root.after_cancel(watchdog)
        self.assertFalse(self.ui._rendering)
        self.assertEqual(len(list(self.directory.glob("Sequence*.png"))), 12)
        self.assertIn("Complete: 12/12", self.ui.status.cget("text"))

    def test_write_failure_reports_partial_output_and_no_success(self):
        self.ui.add_shot()
        with patch.object(Image.Image, "save", side_effect=OSError("disk full")), \
                patch.object(app.messagebox, "showerror") as error:
            self._start_render()
            self._drain_render()
        error.assert_called_once()
        self.assertIn("complete frames retained", error.call_args.args[1])
        self.assertFalse(self.ui._rendering)
        self.assertIn("Failed: 0/12", self.ui.status.cget("text"))

    def test_project_load_is_responsive_and_transactional(self):
        project = self.ui.project
        loaded = app.new_project(replace(self.params, angle=45))
        loaded.sequences[0].shots = [app.Shot(replace(self.params)) for _ in range(20)]
        started, release = threading.Event(), threading.Event()
        initial_cards = []
        install = self.ui._install_project

        def installed(*args):
            install(*args)
            initial_cards.append((len(self.ui._card_widgets), len(self.ui._thumbnails)))

        def load(path):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release project loading")
            return loaded

        with patch.object(app, "load_project_file", side_effect=load), \
                patch.object(self.ui, "_install_project", side_effect=installed), \
                patch.object(app.filedialog, "askopenfilename", return_value="example.json"):
            try:
                self.ui.open_project()
                self._wait_until(started.is_set)
                self.assertIs(self.ui.project, project)
                self.assertEqual(str(self.ui.sequence_choice.cget("state")), "disabled")
                self._assert_heartbeat_until_idle(release)
            finally:
                release.set()
        self.assertIs(self.ui.project, loaded)
        self.assertEqual(self.ui.p.angle, 45)
        self.assertEqual(initial_cards, [(8, 0)])
        self._wait_until(lambda: len(self.ui._thumbnails) == 20)
        self.assertEqual(len(self.ui._card_widgets), 20)

    def test_cancelled_load_keeps_old_project_until_worker_acknowledges(self):
        original = self.ui.project
        started, release = threading.Event(), threading.Event()

        def load(path):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release project loading")
            return app.new_project(self.params)

        with patch.object(app, "load_project_file", side_effect=load), \
                patch.object(app.filedialog, "askopenfilename", return_value="example.json"):
            try:
                self.ui.open_project()
                self._wait_until(started.is_set)
                self.ui.cancel_sequence_render()
                self.assertTrue(self.ui._rendering)
                self.assertIs(self.ui.project, original)
            finally:
                release.set()
            self._drain_render()
        self.assertIs(self.ui.project, original)
        self.assertEqual(str(self.ui.sequence_choice.cget("state")), "readonly")

    def test_superseded_thumbnail_and_preview_results_are_not_applied(self):
        started, release = threading.Event(), threading.Event()

        def thumbnail(p):
            if p.angle == 0:
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release thumbnail")
            return Image.new("RGB", (90, 90), "red" if p.angle == 0 else "blue")

        with patch.object(app, "_thumbnail_image", side_effect=thumbnail):
            try:
                self.ui.add_shot()
                shot = self.ui.sequence.shots[0]
                self._wait_until(started.is_set)
                self.ui.edit_shot(shot.id)
                self.ui.vars["angle"].set(45)
                self.ui.update_shot()
            finally:
                release.set()
            self._wait_until(lambda: shot.id in self.ui._thumbnails)
            self.assertEqual(ImageTk.getimage(self.ui._thumbnails[shot.id]).getpixel((0, 0))[:3],
                             (0, 0, 255))

        started.clear()
        release.clear()

        def preview(p, width, height, frame):
            if p.angle == 45:
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release preview")
            image = Image.new("RGB", (width, height), "red" if p.angle == 45 else "blue")
            return image, image.copy()

        with patch.object(app, "_preview_images", side_effect=preview):
            try:
                self.ui.update_preview()
                self._wait_until(started.is_set)
                self.ui.vars["angle"].set(90)
            finally:
                release.set()
            self._wait_until(lambda: self.ui._preview_bg is not None
                             and self.ui._preview_bg.getpixel((0, 0)) == (0, 0, 255))

    def test_sequence_and_animation_exports_remain_responsive_during_writes(self):
        self.ui.add_shot()
        original_save = Image.Image.save
        for animation in (False, True):
            started, release = threading.Event(), threading.Event()

            def save(image, *args, **kwargs):
                if not release.is_set():
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release PNG write")
                return original_save(image, *args, **kwargs)

            with self.subTest(animation=animation), patch.object(Image.Image, "save", save), \
                    patch.object(app.filedialog, "asksaveasfilename",
                                 return_value=str(self.directory / "animation.gif")):
                try:
                    if animation:
                        self.ui.save_animation()
                    else:
                        self._start_render()
                    self._wait_until(started.is_set)
                    self._assert_heartbeat_until_idle(release)
                finally:
                    release.set()
                self.assertIn("Complete: 12/12", self.ui.status.cget("text"))

    def test_animation_cancellation_and_gif_failure_restore_controls(self):
        output = self.directory / "animation.gif"
        original_save = Image.Image.save
        saved, release = threading.Event(), threading.Event()

        def delayed_save(image, *args, **kwargs):
            original_save(image, *args, **kwargs)
            saved.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release PNG write")

        with patch.object(app.filedialog, "asksaveasfilename", return_value=str(output)):
            with patch.object(Image.Image, "save", delayed_save):
                try:
                    self.ui.save_animation()
                    self._wait_until(saved.is_set)
                    self.ui.cancel_sequence_render()
                    self.assertTrue(self.ui._rendering)
                finally:
                    release.set()
                self._drain_render()
            self.assertFalse(output.exists())
            self.assertEqual(len(list(self.directory.glob("animation_*.png"))), 1)
            self.assertIn("Cancelled: 1/12", self.ui.status.cget("text"))

            def failing_gif(image, path, *args, **kwargs):
                if str(path).endswith(".gif"):
                    raise OSError("GIF encoding failed")
                return original_save(image, path, *args, **kwargs)

            with patch.object(Image.Image, "save", failing_gif), \
                    patch.object(app.messagebox, "showerror") as error:
                self.ui.save_animation()
                self._drain_render()
            error.assert_called_once()
            self.assertIn("GIF encoding failed", error.call_args.args[1])
            self.assertIn("Failed: 12/12", self.ui.status.cget("text"))
            self.assertEqual(str(self.ui.sequence_choice.cget("state")), "readonly")

    def test_animation_preserves_gif_timing_loop_and_png_frames(self):
        self.no_stripes.stop()
        for crowd_on in (False, True):
            with self.subTest(crowd=crowd_on):
                self.ui.vars["crowd_enabled"].set(crowd_on)
                self.ui.vars["crowd_count"].set(1)
                p = replace(self.ui.read_params())
                output = self.directory / f"animation_{crowd_on}.gif"
                with patch.object(app.filedialog, "asksaveasfilename", return_value=str(output)):
                    self.ui.save_animation()
                    self._drain_render()
                with Image.open(output) as gif:
                    self.assertEqual(gif.size, (16, 16))
                    if crowd_on:
                        self.assertNotIn("loop", gif.info)
                    else:
                        self.assertEqual(gif.info["loop"], 0)
                        self.assertEqual(gif.n_frames, 6)
                        for index in range(gif.n_frames):
                            gif.seek(index)
                            self.assertEqual(gif.info["duration"], 160)
                renderer = app.ShotRenderer(p)
                for index in range(12):
                    with Image.open(output.with_name(f"{output.stem}_{index + 1:04d}.png")) as png:
                        self.assertEqual(png.tobytes(), renderer.frame(index).tobytes())


class ProcessWorkflowTests(unittest.TestCase):
    def test_interaction_preview_and_export_in_real_worker(self):
        spec = app.crowd.CustomPedestrian(
            "petter", ((-2, 1), (-1, 1)), walk_duration=3, pet_dog=True,
            pet_seconds=.5, pet_offset=(.08, -.04))
        p = app.Params(width=64, height=64, crowd_enabled=False, anim_fps=4,
                       anim_cycles=2, custom_pedestrians=(spec,))
        frame = app.interaction_for(p).pet_start
        worker = app._BackgroundWorker()
        try:
            result = worker.submit(
                partial(app._background_call, app._preview_images, (p, 64, 64, frame)), 1
            ).result(timeout=30)
            self.assertIsNone(result.error)
            self.assertEqual(result.value[1].tobytes(),
                             app.ShotRenderer(p, supersample=2).frame(frame).tobytes())
            with tempfile.TemporaryDirectory() as directory:
                base = str(Path(directory) / "interaction")
                result = worker.submit(partial(app._export_animation, p, base), 2).result(timeout=30)
                self.assertIsNone(result.error)
                self.assertEqual(result.value, "Complete")
                with Image.open(f"{base}_{frame + 1:04d}.png") as output:
                    self.assertEqual(output.tobytes(), app.ShotRenderer(p).frame(frame).tobytes())
        finally:
            worker.shutdown(wait=True, cancel_futures=True)

    def test_real_worker_load_preview_exports_errors_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            backgrounds = []
            for color in ("red", "blue"):
                path = directory / f"{color}.png"
                Image.new("RGB", (20, 20), color).save(path)
                backgrounds.append(str(path))
            p = app.Params(width=32, height=24, anim_cycles=1, crowd_enabled=False,
                           dog_enabled=False, background_images=tuple(backgrounds),
                           background_perspective=False, stripe_wobble=0)
            project = app.new_project(p)
            project.sequences[0].shots = [app.Shot(replace(p)), app.Shot(replace(p, width=48))]
            path = directory / "project.json"
            app.save_project_file(project, path)
            root = tk.Tk()
            root.withdraw()
            with patch.object(app, "load_defaults", return_value=replace(p)):
                ui = app.App(root)
            if ui._job is not None:
                root.after_cancel(ui._job)
                ui._job = None

            def wait(predicate):
                expired = []

                def poll():
                    if predicate():
                        root.quit()
                    else:
                        root.after(10, poll)

                def timeout():
                    expired.append(True)
                    root.quit()

                watchdog = root.after(15000, timeout)
                root.after(10, poll)
                root.mainloop()
                root.after_cancel(watchdog)
                self.assertFalse(expired, "Worker process timed out")

            try:
                with patch.object(app.messagebox, "showerror") as error:
                    with patch.object(app.filedialog, "askopenfilename", return_value=str(path)):
                        ui.open_project()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui.project, project)
                    wait(lambda: len(ui._thumbnails) == 2 and ui._future is None)

                    with patch.object(app, "render", side_effect=AssertionError("UI thread rendered")):
                        ui.update_preview()
                        wait(lambda: ui._preview_bg is not None and ui._future is None
                             and ui._pending_preview is None)
                        ui.vars["animate_preview"].set(True)
                        ui._anim_tick()
                        wait(lambda: ui._preview_frame == 1)
                        ui.vars["animate_preview"].set(False)
                        ui._toggle_anim_preview_stop()

                    with patch.object(app.filedialog, "askdirectory", return_value=str(directory)):
                        ui.render_sequence()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui._render_written, 24)
                    self.assertIn("Complete: 24/24", ui.status.cget("text"))
                    self.assertEqual(len(list(directory.glob("Sequence1_shot*.png"))), 24)
                    with Image.open(directory / "Sequence1_shot2.0003.png") as png:
                        self.assertEqual(png.size, (48, 24))
                        self.assertEqual(png.tobytes(), app.ShotRenderer(replace(p, width=48)).frame(2).tobytes())

                    gif_path = directory / "animation.gif"
                    with patch.object(app.filedialog, "asksaveasfilename", return_value=str(gif_path)):
                        ui.save_animation()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui._render_written, 12)
                    self.assertTrue(gif_path.exists())
                    error.assert_not_called()

                    cancel_directory = directory / "cancelled"
                    cancel_directory.mkdir()
                    with patch.object(app.filedialog, "askdirectory", return_value=str(cancel_directory)):
                        ui.render_sequence()
                        ui.cancel_sequence_render()
                        wait(lambda: not ui._rendering)
                    self.assertEqual(ui._render_written, 0)
                    self.assertEqual(list(cancel_directory.iterdir()), [])
                    self.assertIn("Cancelled: 0/24", ui.status.cget("text"))

                    original = ui.project
                    invalid = directory / "invalid.json"
                    invalid.write_text("{}")
                    with patch.object(app.filedialog, "askopenfilename", return_value=str(invalid)):
                        ui.open_project()
                        wait(lambda: not ui._rendering)
                    error.assert_called_once()
                    self.assertIs(ui.project, original)
                    self.assertEqual(str(ui.sequence_choice.cget("state")), "readonly")

                    Image.new("RGB", (20, 20), "green").save(backgrounds[0])
                    ui._install_project(app.load_project_file(path), path)
                    refreshed = directory / "refreshed"
                    refreshed.mkdir()
                    with patch.object(app.filedialog, "askdirectory", return_value=str(refreshed)):
                        ui.render_sequence()
                        wait(lambda: not ui._rendering)
                    with Image.open(refreshed / "Sequence1_shot1.0001.png") as png:
                        self.assertEqual(png.getpixel((0, 0)), (0, 128, 0))
            finally:
                root.destroy()
                ui._executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
