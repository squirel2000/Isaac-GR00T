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
    
    # Basic validation
    if frame_count <= 0:
        print(f"Warning: Video {video_path} reported {frame_count} frames. Treating as invalid.", file=sys.stderr)
        return fps, 0, width, height # Return 0 frames but other info if available
    if fps <= 0:
        print(f"Warning: Video {video_path} reported {fps} FPS. This might be unreliable.", file=sys.stderr)
        # Allow processing but FPS might be unreliable if not overridden
    
    return fps, frame_count, width, height

def generate_metadata_script(
    dataset_root_path: Path,
    output_meta_dir_path: Path,
    codebase_version: str,
    robot_type: str,
    task_description: str,
    total_tasks_in_dataset: int,
    data_path_template_str: str,
    video_path_template_str: str,
    video_key: str,
    chunk_size_val: int,
    fps_override_val: float | None,
    splits_definition_dict: dict,
    features_schema_dict: dict
):
    """
    Generates episodes.jsonl and info.json for a dataset.
    """
    output_meta_dir_path.mkdir(parents=True, exist_ok=True)

    episodes_data = []
    total_frames_sum = 0
    
    detected_fps_vals = []
    detected_heights = []
    detected_widths = []

    # Discover episodes: DATASET_ROOT/videos/chunk-*/observation.images.{video_key}/episode_*.mp4
    video_files_glob_pattern = str(dataset_root_path / "videos" / "chunk-*" / f"observation.images.{video_key}" / "episode_*.mp4")
    discovered_video_files = sorted(glob.glob(video_files_glob_pattern))

    if not discovered_video_files:
        print(f"Error: No video files found matching pattern: {video_files_glob_pattern}", file=sys.stderr)
        print("Please ensure your dataset has videos in the expected structure, e.g.,", file=sys.stderr)
        print(f"{dataset_root_path}/videos/chunk-000/observation.images.{video_key}/episode_000000.mp4", file=sys.stderr)
        return False

    print(f"Found {len(discovered_video_files)} potential video files.")

    processed_episode_count = 0
    for video_file_path_str in discovered_video_files:
        video_file_path = Path(video_file_path_str)
        episode_idx = processed_episode_count 

        fps, frame_count, width, height = get_video_metadata(video_file_path)

        if frame_count is None or frame_count <= 0:
            print(f"Skipping episode (would-be index {episode_idx}) due to video error or zero/invalid frames: {video_file_path}", file=sys.stderr)
            continue 

        if fps is not None and fps > 0: detected_fps_vals.append(fps)
        if height is not None and height > 0: detected_heights.append(height)
        if width is not None and width > 0: detected_widths.append(width)
        
        episodes_data.append({
            "episode_index": episode_idx,
            "task": task_description,
            "length": frame_count
        })
        total_frames_sum += frame_count
        processed_episode_count += 1
    
    if not episodes_data:
        print("Error: No valid episodes processed. Cannot generate metadata files.", file=sys.stderr)
        return False
        
    total_episodes = len(episodes_data)
    print(f"Successfully processed {total_episodes} episodes.")

    # Write episodes.jsonl
    episodes_jsonl_path = output_meta_dir_path / "episodes.jsonl"
    with open(episodes_jsonl_path, 'w') as f:
        for episode_info in episodes_data:
            f.write(json.dumps(episode_info) + '\n')
    print(f"Generated {episodes_jsonl_path}")

    # Determine final FPS, height, width for info.json
    final_fps = fps_override_val
    if final_fps is None:
        if detected_fps_vals:
            unique_fps = sorted(list(set(round(f, 2) for f in detected_fps_vals)))
            if len(unique_fps) > 1:
                print(f"Warning: Multiple FPS values detected: {unique_fps}. Using the first one: {unique_fps[0]}", file=sys.stderr)
            final_fps = unique_fps[0]
        else:
            final_fps = 30.0 
            print(f"Warning: No FPS detected from videos and no override. Defaulting to {final_fps} FPS.", file=sys.stderr)
    
    final_video_height = None
    if detected_heights:
        unique_heights = sorted(list(set(detected_heights)))
        if len(unique_heights) > 1:
            print(f"Warning: Multiple video heights detected: {unique_heights}. Using the first one: {unique_heights[0]}", file=sys.stderr)
        final_video_height = unique_heights[0]
    else:
        final_video_height = 480 
        print(f"Warning: No video height detected. Defaulting to {final_video_height}.", file=sys.stderr)

    final_video_width = None
    if detected_widths:
        unique_widths = sorted(list(set(detected_widths)))
        if len(unique_widths) > 1:
            print(f"Warning: Multiple video widths detected: {unique_widths}. Using the first one: {unique_widths[0]}", file=sys.stderr)
        final_video_width = unique_widths[0]
    else:
        final_video_width = 640
        print(f"Warning: No video width detected. Defaulting to {final_video_width}.", file=sys.stderr)

    total_chunks = math.ceil(total_episodes / chunk_size_val) if chunk_size_val > 0 else 1

    final_splits = {}
    if splits_definition_dict:
        for split_name, split_range in splits_definition_dict.items():
            final_splits[split_name] = split_range.replace("N", str(total_episodes))
    else:
        final_splits["train"] = f"0:{total_episodes}"

    current_features_schema = json.loads(json.dumps(features_schema_dict)) 
    
    image_feature_key = f"observation.images.{video_key}"
    if image_feature_key in current_features_schema:
        current_features_schema[image_feature_key]["shape"] = [
            current_features_schema[image_feature_key]["shape"][0], 
            final_video_height,
            final_video_width
        ]
        current_features_schema[image_feature_key]["info"]["video.fps"] = float(final_fps)
        current_features_schema[image_feature_key]["info"]["video.height"] = final_video_height
        current_features_schema[image_feature_key]["info"]["video.width"] = final_video_width
    else:
        print(f"Warning: Image feature key '{image_feature_key}' not found in features schema. Video metadata in info.json might be incomplete.", file=sys.stderr)

    info_content = {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames_sum,
        "total_tasks": total_tasks_in_dataset,
        "total_videos": total_episodes,
        "total_chunks": total_chunks,
        "chunks_size": chunk_size_val,
        "fps": float(final_fps), 
        "splits": final_splits,
        "data_path": data_path_template_str,
        "video_path": video_path_template_str.replace("{video_key}", video_key),
        "features": current_features_schema
    }

    info_json_path = output_meta_dir_path / "info.json"
    with open(info_json_path, 'w') as f:
        json.dump(info_content, f, indent=4)
    print(f"Generated {info_json_path}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata files (info.json, episodes.jsonl) for a robot dataset. \n"
                    "The script expects video files to be structured as: \n"
                    "<dataset_root>/videos/chunk-<CHUNK_ID>/observation.images.<video_key>/episode-<EPISODE_ID>.mp4",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dataset_root", type=str, help="Path to the root directory of the dataset (Required).")
    
    parser.add_argument("--codebase_version", type=str, default="v2.0", help="Codebase version (default: v2.0).")
    parser.add_argument("--robot_type", type=str, default="Unitree_G1", help="Robot type (default: Unitree_G1).")
    parser.add_argument("--task_description", type=str, default="pick and sort a red or blue can",
                        help="Task description for episodes.jsonl (default: 'pick and sort a red or blue can').")
    parser.add_argument("--total_tasks", type=int, default=1,
                        help="Total number of unique tasks in the dataset (default: 1).")
    
    parser.add_argument("--data_path_template", type=str, 
                        default="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                        help="Template for data file paths.")
    parser.add_argument("--video_key", type=str, default="camera", 
                        help="The '{video_key}' part in video_path_template (e.g., 'camera') (default: 'camera').")
    parser.add_argument("--video_path_template", type=str,
                        default="videos/chunk-{episode_chunk:03d}/observation.images.{video_key}/episode_{episode_index:06d}.mp4",
                        help="Template for video file paths.")
    
    parser.add_argument("--chunk_size", type=int, default=1000, 
                        help="Number of episodes per chunk (default: 1000).")
    parser.add_argument("--fps_override", type=float, default=None, 
                        help="Override FPS (e.g., 30.0). If None (default), detects from video.")
    
    parser.add_argument("--splits_json", type=str, default='{"train": "0:N"}',
                        help="JSON string defining dataset splits. 'N' is replaced by total episodes. \n"
                             "(default: '{\"train\": \"0:N\"}')")
    
    parser.add_argument("--joint_names_json", type=str, default=None,
                        help="JSON string of a list of joint names. If None, uses a default list. \n"
                             "Example: '[\"joint1\", \"joint2\"]'")

    args = parser.parse_args()

    dataset_root_p = Path(args.dataset_root)
    output_meta_dir_p = dataset_root_p / "meta"

    if args.joint_names_json:
        try:
            joint_names = json.loads(args.joint_names_json)
            if not isinstance(joint_names, list) or not all(isinstance(j, str) for j in joint_names):
                raise ValueError("Joint names must be a list of strings.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Invalid JSON for --joint_names_json: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        joint_names = [
            "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
            "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
            "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
            "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
            "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
            "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
            "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
            "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1"
        ]
    
    features_schema = {
        "observation.state": {
            "dtype": "float32", "shape": [len(joint_names)], "names": [joint_names]
        },
        "action": {
            "dtype": "float32", "shape": [len(joint_names)], "names": [joint_names]
        },
        f"observation.images.{args.video_key}": {
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

    try:
        splits_dict = json.loads(args.splits_json)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON string for --splits_json: {args.splits_json}", file=sys.stderr)
        print("Using default split: {'train': '0:N'}", file=sys.stderr)
        splits_dict = {"train": "0:N"}

    success = generate_metadata_script(
        dataset_root_path=dataset_root_p,
        output_meta_dir_path=output_meta_dir_p,
        codebase_version=args.codebase_version,
        robot_type=args.robot_type,
        task_description=args.task_description,
        total_tasks_in_dataset=args.total_tasks,
        data_path_template_str=args.data_path_template,
        video_path_template_str=args.video_path_template,
        video_key=args.video_key,
        chunk_size_val=args.chunk_size,
        fps_override_val=args.fps_override,
        splits_definition_dict=splits_dict,
        features_schema_dict=features_schema
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    # Usage example:
    # ./generate_dataset_meta.py /home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T/demo_data/G1_CanSort_Dataset
    main()
