import json
import logging
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lerobot_async_inference.camera_image_logger import CameraImageWriter
from lerobot_async_inference.configs import RobotClientConfig
from lerobot_async_inference.image_transport import encode_observation_images


class CameraImageLoggerTest(unittest.TestCase):
    def test_config_accepts_camera_capture_cli_fields(self) -> None:
        config = RobotClientConfig(
            robot=object(),
            actions_per_chunk=1,
            save_camera_images=True,
            camera_image_log_dir="logs/test-capture",
            camera_image_save_every_n=2,
        )
        self.assertTrue(config.save_camera_images)
        self.assertEqual(config.camera_image_save_every_n, 2)
        with self.assertRaisesRegex(ValueError, "camera_image_save_every_n"):
            RobotClientConfig(
                robot=object(), actions_per_chunk=1, camera_image_save_every_n=0
            )

    def test_writer_saves_transmitted_jpegs_and_manifest(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        observation, _ = encode_observation_images(
            {"camera1": image, "camera2": image, "camera3": image},
            ("camera1", "camera2", "camera3"),
            1.0,
            True,
        )

        with tempfile.TemporaryDirectory() as directory:
            writer = CameraImageWriter(
                directory,
                ("camera1", "camera2", "camera3"),
                1,
                logging.getLogger("camera-image-writer-test"),
            )
            self.assertTrue(writer.submit(observation, wall_time=123.5, timestep=25))
            writer.close()

            manifest = [
                json.loads(line)
                for line in (Path(directory) / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["timestep"], 25)
            self.assertEqual(
                set(manifest[0]["files"]), {"camera1", "camera2", "camera3"}
            )
            for relative_path in manifest[0]["files"].values():
                saved = cv2.imread(str(Path(directory) / relative_path))
                self.assertIsNotNone(saved)
                self.assertEqual(saved.shape, image.shape)

    def test_writer_honors_save_every_n(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            writer = CameraImageWriter(
                directory,
                ("camera1",),
                2,
                logging.getLogger("camera-image-writer-test"),
            )
            self.assertTrue(writer.submit({"camera1": image}, wall_time=1.0, timestep=1))
            self.assertFalse(writer.submit({"camera1": image}, wall_time=2.0, timestep=2))
            self.assertTrue(writer.submit({"camera1": image}, wall_time=3.0, timestep=3))
            writer.close()

            records = (Path(directory) / "manifest.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
