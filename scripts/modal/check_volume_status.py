"""
Check Modal volume upload status.

Prints:
  - Number of files in volume
  - Number of complete batches (batch_size=5)
  - Remaining files in incomplete batch

Usage:
    /home/agrov/gram/bin/python scripts/check_volume_status.py
"""

import modal

volume = modal.Volume.from_name("gram-data", create_if_missing=True)

print("Checking Modal volume 'gram-data'...\n")

try:
    files = volume.listdir("/")
    file_count = len(files)
    batch_size = 5
    complete_batches = file_count // batch_size
    remaining_files = file_count % batch_size

    print(f"Total files uploaded: {file_count}")
    print(f"Complete batches (batch_size={batch_size}): {complete_batches} (batches 0-{complete_batches-1})")
    if remaining_files > 0:
        print(f"Incomplete batch #{complete_batches}: {remaining_files}/{batch_size} files")
    else:
        print("All batches complete!")

except Exception as e:
    print(f"Error checking volume: {e}")
