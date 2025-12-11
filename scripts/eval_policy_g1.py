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

import warnings

import numpy as np
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

python scripts/eval_policy.py --host localhost --port 5555 --plot
    --modality_keys right_arm right_hand
    --steps 250
    --trajs 1000
    --action_horizon 16
    --video_backend decord
    --dataset_path demo_data/robot_sim.PickNPlace/
    --embodiment_tag gr1
    --data_config gr1_arms_waist
provide --model_path to load up the model checkpoint in this script.
"""

if __name__ == "__main__":
    # Fixed configuration for G1 Block Stacking evaluation
    class EvalConfig:
        host: str = "localhost"
        port: int = 5555
        plot: bool = True # Enabled plotting
        modality_keys: list[str] = ["right_arm", "right_hand"]
        data_config: str = "g1_can_pick_and_sort" # g1_block_stacking
        steps: int = 500  # 4 * 100 steps for movements + 2 * 50 steps for grasping
        trajs: int = 1
        action_horizon: int = 16
        video_backend: str = "torchvision_av" # Changed for G1 compatibility
        dataset_path: str = "demo_data/G1_CanSorting_Dataset" # Fixed for G1
        embodiment_tag: str = "new_embodiment" # Fixed for G1
        model_path: str = "output/G1_CanSorting_Dataset/" # Fixed for G1 output directory
        denoising_steps: int = 4

    config = EvalConfig()

    data_config = DATA_CONFIG_MAP[config.data_config]
    if config.model_path is not None:
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=config.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=config.host, port=config.port)

    all_gt_actions = []
    all_pred_actions = []

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality,
        video_backend=config.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=config.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", config.modality_keys)

    all_mse = []
    for traj_id in range(config.trajs):
        print("Running trajectory:", traj_id)
        mse = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=config.modality_keys,
            steps=config.steps,
            action_horizon=config.action_horizon,
            plot=config.plot,
        )
        print("MSE:", mse)
        all_mse.append(mse)
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Done")
    exit()
