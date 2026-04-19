"""
List all files in Modal volume and find extra/missing files.

Usage:
    /home/agrov/gram/bin/python scripts/list_volume_files.py
"""

import modal
from pathlib import Path

volume = modal.Volume.from_name("gram-data", create_if_missing=True)
local_data_dir = Path("/home/agrov/iclr-2026/data")

print("Checking Modal volume 'gram-data'...\n")

try:
    volume_files = set(volume.listdir("/"))
    local_files = set(f.name for f in local_data_dir.glob("*.npz"))

    print(f"Files in volume:  {len(volume_files)}")
    print(f"Files locally:    {len(local_files)}")

    # Find files in volume but not local
    extra = volume_files - local_files
    if extra:
        print(f"\n⚠️  Extra files in volume ({len(extra)}):")
        for f in sorted(extra):
            print(f"  - {f}")

    # Find files local but not in volume
    missing = local_files - volume_files
    if missing:
        print(f"\n❌ Missing from volume ({len(missing)}):")
        for f in sorted(missing)[:10]:  # show first 10
            print(f"  - {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if not extra and not missing:
        print("\n✅ All files match!")

except Exception as e:
    print(f"Error: {e}")
