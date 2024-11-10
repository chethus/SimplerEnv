from collections import defaultdict
from typing import Optional
import requests
import json_numpy
json_numpy.patch()

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from transforms3d.euler import euler2axangle

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, (128, 128), method="lanczos3", antialias=True)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

def get_preprocessed_image(full_image, resize_size):
    """
    Preprocess the image the exact same way that the Berkeley Bridge folks did it
    to minimize distribution shift.
    NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
                    resized up to a different resolution by some models. This is just so that we're in-distribution
                    w.r.t. the original preprocessing at train time.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    full_image = resize_image(full_image, resize_size)
    return full_image

class OpenVLAInference:
    def __init__(
        self,
        agent_host: str = None,
        agent_port: int = 8000,
        policy_setup: str = "widowx_bridge",
    ) -> None:
        self.agent_host = agent_host
        self.agent_port = agent_port

        self.task_description = None

        self.policy_setup = policy_setup
        if self.policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig"
        elif self.policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data"
        else:
            raise NotImplementedError()

    def reset(self, task_description: str) -> None:
        self.task_description = task_description if task_description is not None else ""

    def step(self, image: np.ndarray, task_description: Optional[str] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed
                self.reset(task_description)
        
        assert image.dtype == np.uint8
        image =  get_preprocessed_image(image, 224)

        response = requests.post(
            f"http://{self.agent_host}:{self.agent_port}/act",
            json={
                "observation": {"full_image": image},
                "instruction": self.task_description,
                "unnorm_key": "bridge_orig",
                "timestep": 0,
            }
        ).json()["action"]

        raw_action = {
            "world_vector": response[:3],
            "rotation_delta": response[3:6],
            "open_gripper": response[6:],
        }
        action = raw_action

        action = {
            "world_vector": raw_action["world_vector"],
        }
        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle

        # Taken from OctoInference class
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def visualize_epoch(self, predicted_raw_actions, images, save_path):
        images = [self._resize_image(image) for image in images]
        predicted_action_name_to_values_over_time = defaultdict(list)
        figure_layout = [
            "terminate_episode_0",
            "terminate_episode_1",
            "terminate_episode_2",
            "world_vector_0",
            "world_vector_1",
            "world_vector_2",
            "rotation_delta_0",
            "rotation_delta_1",
            "rotation_delta_2",
            "gripper_closedness_action_0",
        ]
        action_order = [
            "terminate_episode",
            "world_vector",
            "rotation_delta",
            "gripper_closedness_action",
        ]

        for i, action in enumerate(predicted_raw_actions):
            for action_name in action_order:
                for action_sub_dimension in range(action[action_name].shape[0]):
                    # print(action_name, action_sub_dimension)
                    title = f"{action_name}_{action_sub_dimension}"
                    predicted_action_name_to_values_over_time[title].append(
                        predicted_raw_actions[i][action_name][action_sub_dimension]
                    )

        figure_layout = [["image"] * len(figure_layout), figure_layout]

        plt.rcParams.update({"font.size": 12})

        stacked = tf.concat(tf.unstack(images[::3], axis=0), 1)

        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        for i, (k, v) in enumerate(predicted_action_name_to_values_over_time.items()):
            axs[k].plot(predicted_action_name_to_values_over_time[k], label="predicted action")
            axs[k].set_title(k)
            axs[k].set_xlabel("Time in one episode")

        axs["image"].imshow(stacked.numpy())
        axs["image"].set_xlabel("Time in one episode (subsampled)")

        plt.legend()
        plt.savefig(save_path)
