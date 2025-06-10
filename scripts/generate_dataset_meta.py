#!/usr/bin/env python3

import os
import json
import glob
import math
import cv2  # OpenCV for video processing
from pathlib import Path
import argparse
import sys

def get_video_metadata(video_path: Path):
    """
    Extracts metadata (fps, frame_count, width, height) from a video file.
    Returns None for any value if extraction fails or video is invalid.
    """
    if not video_path.exists() or not video_path.is_file():
        print(f"Warning: Video file not found or is not a file: {video_path}", file=sys.stderr)
        return None, None, None, None
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video file {video_path}", file=sys.stderr)
        return None, None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Store original values for more informative warnings
    original_fps, original_width, original_height = fps, width, height

    # Validate frame_count first, as this determines if the video is usable for an episode
    if frame_count <= 0:
        print(f"Warning: Video {video_path} reported {frame_count} frames. Treating as invalid for episode processing.", file=sys.stderr)
        return original_fps, 0, original_width, original_height

    # For videos with valid frame_count, ensure other critical metadata is also valid (positive)
    # If not, set them to None, as they cannot be used for dataset-wide metadata.
    if original_fps <= 0:
        print(f"Warning: Video {video_path} reported FPS {original_fps}. Treating as invalid.", file=sys.stderr)
        fps = None
    if original_width <= 0:
        print(f"Warning: Video {video_path} reported width {original_width}. Treating as invalid.", file=sys.stderr)
        width = None
    if original_height <= 0:
        print(f"Warning: Video {video_path} reported height {original_height}. Treating as invalid.", file=sys.stderr)
        height = None
    
    return fps, frame_count, width, height

def generate_metadata_script(
    dataset_root_path: Path,
    output_meta_dir_path: Path,
    codebase_version_const: str,
    robot_type_const: str,
    task_description_from_file: str,
    total_tasks_from_file: int,
    data_path_template_str: str,
    video_path_template_str: str,
    video_key: str,
    chunk_size_val: int,
    features_schema_dict: dict
):
    """
    Generates episodes.jsonl and info.json for a dataset.
    """
    output_meta_dir_path.mkdir(parents=True, exist_ok=True)

    episodes_data = []
    total_frames_sum = 0

    # Discover episodes: DATASET_ROOT/videos/chunk-*/{video_key}/episode_*.mp4
    video_files_glob_pattern = str(dataset_root_path / "videos" / "chunk-*" / video_key / "episode_*.mp4")
    discovered_video_files = sorted(glob.glob(video_files_glob_pattern))

    if not discovered_video_files:
        print(f"Error: No video files found matching pattern: {video_files_glob_pattern}", file=sys.stderr)
        print("Please ensure your dataset has videos in the expected structure, e.g.,", file=sys.stderr)
        print(f"{dataset_root_path}/videos/chunk-000/{video_key}/episode_000000.mp4", file=sys.stderr)
        return False

    processed_episode_count = 0
    current_fps = current_frame_count = current_width = current_height = None  # Ensure variables are defined
    for video_file_path_str in discovered_video_files:
        video_file_path = Path(video_file_path_str)
        
        # Get metadata for the current video
        current_fps, current_frame_count, current_width, current_height = get_video_metadata(video_file_path)

        if current_frame_count is None or current_frame_count <= 0:
            print(f"Terminated due to error or zero/invalid frames: {video_file_path}", file=sys.stderr)
            return False
        # If this is the first valid video, set and validate the dataset-wide metadata
        if current_fps is None or current_height is None or current_width is None:
            print(f"Error: Invalid FPS/height/width obtained from video: {video_file_path}. Cannot proceed.", file=sys.stderr)
            return False

        episode_idx = processed_episode_count
        episodes_data.append({
            "episode_index": episode_idx,
            "task": task_description_from_file,
            "length": current_frame_count
        })
        total_frames_sum += current_frame_count
        processed_episode_count += 1
    
    total_episodes = len(episodes_data)
    print(f"Successfully processed {total_episodes} episodes with FPS={current_fps}, Height={current_height}, Width={current_width}\n")
    
    # Write episodes.jsonl
    episodes_jsonl_path = output_meta_dir_path / "episodes.jsonl"
    with open(episodes_jsonl_path, 'w') as f:
        for episode_info in episodes_data:
            f.write(json.dumps(episode_info) + '\n')
    print(f"Generated {episodes_jsonl_path}")

    # info.json generation
    total_chunks = math.ceil(total_episodes / chunk_size_val) if chunk_size_val > 0 else 1

    # Simplified splits: all data for training
    final_splits = {"train": f"0:{total_episodes}"}

    current_features_schema = json.loads(json.dumps(features_schema_dict)) 
    image_feature_key = "observation.images.camera" # Hardcoded schema key
    if image_feature_key in current_features_schema and current_height is not None and current_width is not None and current_fps is not None:
        current_features_schema[image_feature_key]["shape"] = [
            current_features_schema[image_feature_key]["shape"][0], 
            current_height,
            current_width
        ]
        current_features_schema[image_feature_key]["info"]["video.fps"] = float(current_fps)
        current_features_schema[image_feature_key]["info"]["video.height"] = current_height
        current_features_schema[image_feature_key]["info"]["video.width"] = current_width
    else:
        print(f"Warning: Image feature key '{image_feature_key}' not found in features schema or video metadata is incomplete. Video metadata in info.json might be incomplete.", file=sys.stderr)

    info_content = {
        "codebase_version": codebase_version_const,
        "robot_type": robot_type_const,
        "total_episodes": total_episodes,
        "total_frames": int(total_frames_sum),
        "total_tasks": total_tasks_from_file,
        "total_videos": total_episodes,
        "total_chunks": total_chunks,
        "chunks_size": chunk_size_val,
        "fps": float(current_fps) if current_fps is not None else 0.0, 
        "splits": final_splits,
        "data_path": data_path_template_str,
        "video_path": video_path_template_str, # The template is now fully qualified
        "features": current_features_schema
    }

    info_json_path = output_meta_dir_path / "info.json"
    with open(info_json_path, 'w') as f:
        json.dump(info_content, f, indent=4)
    print(f"Generated {info_json_path}")
    return True

def load_tasks_jsonl(tasks_jsonl_path: Path) -> tuple[str | None, int | None]:
    """Loads task description and count from tasks.jsonl."""
    default_task_desc = "pick and sort a red or blue can" # Fallback if "task" key is missing
    if not tasks_jsonl_path.is_file():
        print(f"Error: Tasks file '{tasks_jsonl_path}' not found.", file=sys.stderr)
        print("Please create this file with each line as a JSON object, e.g.: {\"task_index\": 0, \"task\": \"pick and sort a red or blue can\"}", file=sys.stderr)
        return None, None

    tasks_data_list = []
    try:
        with open(tasks_jsonl_path, 'r') as f_tasks:
            for line_idx, line in enumerate(f_tasks):
                line_content = line.strip()
                if line_content:
                    try:
                        tasks_data_list.append(json.loads(line_content))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode JSON from {tasks_jsonl_path} line {line_idx + 1}: '{line_content}' - {e}", file=sys.stderr)
        
        if not tasks_data_list:
            print(f"Error: {tasks_jsonl_path} is empty or contains no valid JSON. Cannot determine task description or count.", file=sys.stderr)
            return None, None
        
        task_description = tasks_data_list[0].get("task", default_task_desc)
        num_tasks = len(tasks_data_list)
        print(f"Loaded {num_tasks} task(s) from {tasks_jsonl_path}.")
        print(f"Using task description for episodes: '{task_description}'\n")
        return task_description, num_tasks
    except Exception as e:
        print(f"Error reading or processing {tasks_jsonl_path}: {e}", file=sys.stderr)
        return None, None

def main():
    DEFAULT_CODEBASE_VERSION = "v2.0"
    DEFAULT_ROBOT_TYPE = "Unitree_G1"
    DEFAULT_DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    DEFAULT_VIDEO_KEY = "observation.images.camera"
    DEFAULT_VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/observation.images.camera/episode_{episode_index:06d}.mp4"

    DEFAULT_JOINT_NAMES = [
        "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
        "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
        "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
        "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
        "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
        "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
        "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
        "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1"
    ]
    parser = argparse.ArgumentParser(
        description="Generate metadata files (info.json, episodes.jsonl) for a robot dataset. \n"
                    "The script expects video files to be structured as: \n"
                    "<dataset_root>/videos/chunk-<CHUNK_ID>/observation.images.<video_key>/episode-<EPISODE_ID>.mp4",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dataset_root", type=str, help="Path to the root directory of the dataset (Required).")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Number of episodes per chunk (default: 1000).")

    args = parser.parse_args()
    dataset_root_p = Path(args.dataset_root)
    output_meta_dir_p = dataset_root_p / "meta"

    # Read task description and total tasks from meta/tasks.jsonl
    tasks_jsonl_path = output_meta_dir_p / "tasks.jsonl"
    task_description_for_script, num_tasks_for_script = load_tasks_jsonl(tasks_jsonl_path)
    if task_description_for_script is None or num_tasks_for_script is None:
        sys.exit(1)

    features_schema = {
        "observation.state": {
            "dtype": "float32", "shape": [len(DEFAULT_JOINT_NAMES)], "names": [DEFAULT_JOINT_NAMES]
        },
        "action": {
            "dtype": "float32", "shape": [len(DEFAULT_JOINT_NAMES)], "names": [DEFAULT_JOINT_NAMES]
        },
        "observation.images.camera": { # Hardcoded schema key
            "dtype": "video",
            "shape": [3, None, None], # H, W auto-filled
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": None, # Auto-filled
                "video.height": None, # Auto-filled
                "video.width": None, # Auto-filled
                "video.channels": 3, 
                "video.codec": "mp4v", 
                "video.pix_fmt": "yuv420p", 
                "video.is_depth_map": False,
                "has_audio": False 
            }
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None}, 
        "task_index": {"dtype": "int64", "shape": [1], "names": None} 
    }

    success = generate_metadata_script(
        dataset_root_path=dataset_root_p,
        output_meta_dir_path=output_meta_dir_p,
        codebase_version_const=DEFAULT_CODEBASE_VERSION,
        robot_type_const=DEFAULT_ROBOT_TYPE,
        task_description_from_file=task_description_for_script,
        total_tasks_from_file=num_tasks_for_script,
        data_path_template_str=DEFAULT_DATA_PATH_TEMPLATE,
        video_path_template_str=DEFAULT_VIDEO_PATH_TEMPLATE,
        video_key=DEFAULT_VIDEO_KEY,
        chunk_size_val=args.chunk_size,
        features_schema_dict=features_schema
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    # Usage example:
    # python scripts/generate_dataset_meta.py demo_data/G1_CanSorting_Dataset
    main()
