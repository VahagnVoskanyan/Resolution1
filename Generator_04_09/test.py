#!/usr/bin/env python3
import argparse
from pathlib import Path


 # Deletes the empty files
def delete_empty_jsonl_files(directory: Path):
    """
    Delete all empty .jsonl files in the given directory.
    Prints the path of each file it deletes.
    """
    for file in directory.glob("*.jsonl"):
        if file.is_file() and file.stat().st_size == 0:
            print(f"Deleting empty file: {file}")
            file.unlink()

def main():
    parser = argparse.ArgumentParser(
        description="Delete empty .jsonl files from a directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="Res_Pairs - Copy",
        help="Directory to scan (default: current directory)."
    )
    args = parser.parse_args()
    dir_path = Path(args.directory).resolve()
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a directory.")
        return
    delete_empty_jsonl_files(dir_path)

if __name__ == "__main__":
    main()
