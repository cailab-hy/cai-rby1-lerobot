import pickle  # nosec
import unittest
from dataclasses import replace

import numpy as np

from lerobot_async_inference.configs import RobotClientConfig
from lerobot_async_inference.helpers import TimedObservation
from lerobot_async_inference.image_transport import (
    decode_observation_images,
    encode_observation_images,
)


CAMERA_KEYS = ("front", "left", "right")


def make_color_test_image(height: int = 480, width: int = 640) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    third = width // 3
    image[:, :third] = (255, 0, 0)
    image[:, third : 2 * third] = (0, 255, 0)
    image[:, 2 * third :] = (0, 0, 255)
    return image


class ImageTransportTest(unittest.TestCase):
    def _round_trip(self, scale: float, jpeg_compression: bool):
        images = {key: make_color_test_image() for key in CAMERA_KEYS}
        state = np.arange(8, dtype=np.float32)
        raw_observation = {
            **images,
            "state_vector": state,
            "joint_position": 0.25,
            "task": "pick up the red block",
        }
        original = TimedObservation(
            timestamp=123.5,
            timestep=17,
            observation=raw_observation,
            must_go=True,
        )

        if scale == 1.0 and not jpeg_compression:
            outbound = original
            self.assertIs(outbound.get_observation(), raw_observation)
        else:
            encoded, encode_stats = encode_observation_images(
                raw_observation,
                CAMERA_KEYS,
                scale,
                jpeg_compression,
            )
            outbound = replace(original, observation=encoded)
            self.assertEqual(encode_stats.image_count, len(CAMERA_KEYS))
            self.assertLess(encode_stats.transport_bytes, encode_stats.original_bytes)
            self.assertIs(encoded["state_vector"], state)
            self.assertIsInstance(encoded["front"], dict)
            self.assertEqual(encoded["front"]["original_shape"], (480, 640, 3))
            self.assertEqual(
                encoded["front"]["transport_shape"],
                (round(480 * scale), round(640 * scale), 3),
            )
            self.assertEqual(encoded["front"]["channel_order"], "RGB")
            if jpeg_compression:
                self.assertEqual(encoded["front"]["encoding"], "jpeg")
                self.assertIsInstance(encoded["front"]["data"], bytes)
            else:
                self.assertEqual(encoded["front"]["encoding"], "raw_resized")
                self.assertIsInstance(encoded["front"]["data"], np.ndarray)
            self.assertLess(len(pickle.dumps(outbound)), len(pickle.dumps(original)))

        inbound = pickle.loads(pickle.dumps(outbound))  # nosec
        decoded, _ = decode_observation_images(inbound.get_observation())
        inbound.observation = decoded

        self.assertEqual(inbound.timestamp, original.timestamp)
        self.assertEqual(inbound.timestep, original.timestep)
        self.assertEqual(inbound.must_go, original.must_go)
        self.assertEqual(inbound.get_observation()["task"], raw_observation["task"])
        self.assertEqual(inbound.get_observation()["joint_position"], 0.25)
        np.testing.assert_array_equal(inbound.get_observation()["state_vector"], state)

        for camera_key in CAMERA_KEYS:
            image = inbound.get_observation()[camera_key]
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (480, 640, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.ndim, 3)

            # Sample well inside each solid-color region. JPEG is lossy, so
            # channel dominance is a more robust RGB/BGR check than equality.
            red = image[240, 80]
            green = image[240, 320]
            blue = image[240, 560]
            self.assertGreater(int(red[0]), int(red[2]) + 100)
            self.assertGreater(int(green[1]), max(int(green[0]), int(green[2])) + 100)
            self.assertGreater(int(blue[2]), int(blue[0]) + 100)

        # Encoding must never mutate the observation returned by the robot.
        for camera_key in CAMERA_KEYS:
            self.assertIs(raw_observation[camera_key], images[camera_key])
            self.assertIsInstance(raw_observation[camera_key], np.ndarray)

    def test_legacy_transport(self):
        self._round_trip(scale=1.0, jpeg_compression=False)

    def test_resize_only_transport(self):
        self._round_trip(scale=0.5, jpeg_compression=False)

    def test_jpeg_only_transport(self):
        self._round_trip(scale=1.0, jpeg_compression=True)

    def test_resize_and_jpeg_transport(self):
        self._round_trip(scale=0.5, jpeg_compression=True)

    def test_legacy_decode_returns_same_observation(self):
        observation = {"front": make_color_test_image(), "state": np.arange(4)}
        decoded, stats = decode_observation_images(observation)
        self.assertIs(decoded, observation)
        self.assertEqual(stats.image_count, 0)

    def test_invalid_resize_scale_is_rejected_by_client_config(self):
        for scale in (0.0, -0.1, 1.01, float("nan"), float("inf"), "invalid", True):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "image_resize_scale"):
                RobotClientConfig(robot=object(), actions_per_chunk=1, image_resize_scale=scale)


if __name__ == "__main__":
    unittest.main()
