import json
import pandas as pd
from pathlib import Path
import argparse

def update_episode_lengths(dataset_base_path_str: str):
    """
    Updates the 'length' attribute in an episodes.jsonl file for a LeRobot dataset.
    The length is determined by the number of rows in the corresponding Parquet file.

    Args:
        dataset_base_path_str (str): The path to the root of the LeRobot dataset.
    """
    dataset_base_path = Path(dataset_base_path_str)
    episodes_file = dataset_base_path / "meta/episodes.jsonl"
    info_file = dataset_base_path / "meta/info.json"

    if not dataset_base_path.is_dir():
        print(f"Error: Dataset path '{dataset_base_path}' does not exist or is not a directory.")
        return
    if not episodes_file.is_file():
        print(f"Error: Episodes file '{episodes_file}' does not exist.")
        return
    if not info_file.is_file():
        print(f"Error: Info file '{info_file}' does not exist.")
        return

    # 1. Read info.json to get data_path_pattern and chunk_size
    try:
        with open(info_file, 'r') as f:
            info_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{info_file}': {e}")
        return
    
    data_path_pattern = info_data.get("data_path")
    chunk_size = info_data.get("chunks_size")

    if not data_path_pattern or chunk_size is None:
        print("Error: 'data_path' or 'chunks_size' not found in info.json.")
        return

    print(f"Using data_path_pattern: {data_path_pattern}")
    print(f"Using chunk_size: {chunk_size}")

    updated_episodes_data = []
    lines_read = 0
    episodes_updated_count = 0
    errors_encountered = 0

    with open(episodes_file, 'r') as f_episodes:
        for line_num, line in enumerate(f_episodes):
            lines_read += 1
            line_content = line.strip()
            if not line_content: # Skip empty lines
                continue
            try:
                episode_data = json.loads(line_content)
                original_episode_data_str = line_content # For writing back if error

                episode_index = episode_data.get("episode_index")

                if episode_index is None:
                    print(f"Warning: 'episode_index' not found in line {line_num + 1}. Keeping original: {line_content}")
                    updated_episodes_data.append(original_episode_data_str)
                    errors_encountered +=1
                    continue

                episode_chunk = episode_index // chunk_size
                
                relative_parquet_path_str = data_path_pattern.format(episode_chunk=episode_chunk, episode_index=episode_index)
                parquet_file_path = dataset_base_path / relative_parquet_path_str
                
                if not parquet_file_path.is_file():
                    print(f"Warning: Parquet file not found for episode_index {episode_index} at '{parquet_file_path}'. Keeping original length for this episode.")
                    updated_episodes_data.append(original_episode_data_str)
                    errors_encountered +=1
                    continue
                
                df = pd.read_parquet(parquet_file_path)
                actual_length = len(df)
                
                if episode_data.get("length") != actual_length:
                    print(f"Updating episode_index {episode_index}: old length {episode_data.get('length')}, new length {actual_length}")
                    episode_data["length"] = actual_length
                    episodes_updated_count +=1
                
                updated_episodes_data.append(json.dumps(episode_data))

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line {line_num + 1}. Keeping original: {line_content}")
                updated_episodes_data.append(line_content) # Keep original line if it's not empty
                errors_encountered +=1
            except Exception as e:
                print(f"An unexpected error occurred processing line {line_num + 1} (episode_index {episode_data.get('episode_index', 'N/A')}): {e}. Keeping original.")
                updated_episodes_data.append(original_episode_data_str if 'original_episode_data_str' in locals() else line_content)
                errors_encountered +=1

    with open(episodes_file, 'w') as f_episodes_out:
        for updated_line in updated_episodes_data:
            f_episodes_out.write(updated_line + '\n')
    
    print(f"\nProcessing complete for '{episodes_file}'.")
    print(f"Total lines read: {lines_read}")
    print(f"Episode lengths updated: {episodes_updated_count}")
    if errors_encountered > 0:
        print(f"Warnings/Errors encountered: {errors_encountered}. Please review the log.")
    if episodes_updated_count > 0 or errors_encountered > 0:
        print(f"'{episodes_file}' has been rewritten.")
    else:
        print(f"No changes were needed for '{episodes_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update episode lengths in episodes.jsonl based on Parquet file row counts.")
    parser.add_argument("--dataset_path", type=str, help="Path to the LeRobot dataset directory (e.g., demo_data/G1_testing_dataset).")
    args = parser.parse_args()
    
    print(f"Attempting to update episode lengths for dataset: {args.dataset_path}")
    # It's good practice to recommend backing up the file before running.
    print(f"IMPORTANT: This script will modify '{Path(args.dataset_path) / 'meta/episodes.jsonl'}' in place.")
    print("It is recommended to back up this file before proceeding.")
    proceed = input("Do you want to continue? (yes/no): ")
    if proceed.lower() == 'yes':
        update_episode_lengths(args.dataset_path)
    else:
        print("Operation cancelled by the user.")