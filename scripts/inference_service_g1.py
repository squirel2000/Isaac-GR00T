# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import numpy as np

from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

# =====================
# Dataset Constants
# =====================
G1_CHECKPOINTS_DIR = "output/G1_CubeStacking_Dataset_Checkpoints_fft_bs16/"
G1_DATA_CONFIG = "g1_can_pick_and_sort" # Data config for G1 "Can Picking-and-Sorting" / "Cube Stacking" Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory. For G1 CanSorting dataset (pick and sort a can), this might be 'output/G1_CanSorting_Dataset/' or a base model like 'nvidia/GR00T-N1-2B'.",
        default=G1_CHECKPOINTS_DIR,
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment", # Adjusted for G1 BlockStacking
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default=G1_DATA_CONFIG
    )

    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument("--host", type=str, help="Host address for the server.", default="localhost")
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    parser.add_argument("--denoising_steps", type=int, help="Number of denoising steps.", default=4)
    args = parser.parse_args()

    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        import time

        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection
        # Create a policy wrapper
        policy_client = RobotInferenceClient(host=args.host, port=args.port)

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        # Example observation for G1_BlockStacking_Dataset.
        # The actual camera views and resolutions might vary based on your 'g1_block_stacking' data_config.
        # The policy on the server side will handle necessary transforms (e.g., resizing).
        # State dimensions should match your robot's configuration as per modality.json.
        # For G1_BlockStacking_Dataset:
        # - video keys: "video.cam_right_high", "video.cam_left_wrist", "video.cam_right_wrist"
        # - state keys: "state.left_arm" (7), "state.right_arm" (7), "state.left_hand" (7), "state.right_hand" (7)
        # - annotation key: "annotation.human.task_description"
        obs = {
            # Using 'video.cam_right_high' as an example. Shape is (batch, H, W, C)
            "video.camera": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 7),
            "state.right_hand": np.random.rand(1, 7),
            # Task description for G1_testing_dataset (pick and place cube)
            "annotation.human.task_description": ["pick and place a cube"],
        }

        time_start = time.time()
        action = policy_client.get_action(obs)
        print(f"Total time taken to get action from server: {time.time() - time_start} seconds")

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")

    else:
        raise ValueError("Please specify either --server or --client")
