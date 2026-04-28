#!/usr/bin/env python3
"""
Script to reorder series.txt files by moving the first 3696 lines to the end.
Processes all series.txt files recursively in specified directories.
"""

import os
from pathlib import Path

# Directories to process
DIRECTORIES = [
    r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25\input\thermal\series\z_effacement",
    r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25\input\thermal\series\z_report",
]

LINES_TO_MOVE = 3696


def reorder_series_file(file_path: Path) -> None:
    """
    Move the first LINES_TO_MOVE lines to the end of the file.
    
    Args:
        file_path: Path to the series.txt file to reorder
    """
    try:
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if total_lines <= LINES_TO_MOVE:
            print(f"⚠️  {file_path}: File has only {total_lines} lines (expected > {LINES_TO_MOVE}). Skipping.")
            return
        
        # Extract first LINES_TO_MOVE lines and remaining lines
        first_lines = lines[:LINES_TO_MOVE]
        remaining_lines = lines[LINES_TO_MOVE:]
        
        # Reorder: remaining lines + first lines
        reordered_lines = remaining_lines + first_lines
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(reordered_lines)
        
        print(f"✅ {file_path}: Reordered successfully ({total_lines} lines)")
        
    except Exception as e:
        print(f"❌ {file_path}: Error - {e}")


def process_directories(directories: list) -> None:
    """
    Find and process all series.txt files in the specified directories.
    
    Args:
        directories: List of directory paths to search
    """
    total_files = 0
    processed_files = 0
    
    for directory in directories:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"❌ Directory not found: {directory}")
            continue
        
        print(f"\n🔍 Searching in: {directory}")
        
        # Find all series.txt files recursively
        series_files = list(dir_path.rglob("series.txt"))
        
        if not series_files:
            print(f"   No series.txt files found in {directory}")
            continue
        
        print(f"   Found {len(series_files)} series.txt file(s)")
        
        for series_file in series_files:
            total_files += 1
            reorder_series_file(series_file)
            processed_files += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: Processed {processed_files}/{total_files} file(s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("🚀 Starting series.txt reordering process...")
    print(f"   Moving first {LINES_TO_MOVE} lines to end of each file")
    print(f"{'='*70}\n")
    
    process_directories(DIRECTORIES)
    
    print("\n✨ Done!")
