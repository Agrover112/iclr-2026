"""
Upload local data to Modal volume (run once).

Uploads:
  - data/*.npz       — raw simulation files (~14 GB)
  - data/*_feat.pt   — precomputed features (run scripts/precompute_features.py first)

Usage:
    python scripts/upload_data_modal.py

    # Dry run — see what would be uploaded without uploading
    python scripts/upload_data_modal.py --dry-run

    # Re-upload even if files already exist
    python scripts/upload_data_modal.py --overwrite
"""

import argparse
import modal
from pathlib import Path
from tqdm import tqdm

LOCAL_DATA_DIR = Path("/home/agrov/iclr-2026/data")
volume = modal.Volume.from_name("gram-data", create_if_missing=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--npz-only", action="store_true", help="Upload only .npz files, skip feature caches.")
    parser.add_argument("--individual", action="store_true", help="Upload files one-by-one (slower but can skip individual files).")
    parser.add_argument("--skip-n", type=int, default=0, help="Skip first N files (individual mode) or batches (batch mode).")
    args = parser.parse_args()

    files = sorted(LOCAL_DATA_DIR.glob("*.npz"))
    if not args.npz_only:
        files += sorted(LOCAL_DATA_DIR.glob("*_feat.pt"))

    if not files:
        print(f"No files found in {LOCAL_DATA_DIR}")
        return

    npz_count = sum(1 for f in files if f.suffix == ".npz")
    feat_count = sum(1 for f in files if f.name.endswith("_feat.pt"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6

    print(f"Found {npz_count} NPZ files + {feat_count} feature cache files")
    print(f"Total size: {total_mb:.0f} MB")

    if args.dry_run:
        print("\nDry run — not uploading. Remove --dry-run to upload.")
        return

    # Get all existing files in volume
    print("Checking which files exist in volume...\n")
    try:
        existing = set(volume.listdir("/"))
    except Exception:
        existing = set()

    # Filter out already-uploaded files (unless --overwrite)
    original_count = len(files)
    if not args.overwrite:
        files = [f for f in files if f.name not in existing]
        if len(files) < original_count:
            skipped = original_count - len(files)
            print(f"Skipping {skipped} already-uploaded files.")

    if not files:
        print("All files already uploaded!")
        return

    # Apply --skip-n
    if args.skip_n > 0:
        if args.individual:
            print(f"Skipping first {args.skip_n} files (--skip-n).\n")
            files = files[args.skip_n:]
        else:
            batch_size = 5
            print(f"Skipping first {args.skip_n} batches (--skip-n).\n")
            files = files[args.skip_n * batch_size:]

    if not files:
        print("All files skipped!")
        return

    # Upload files
    uploaded = 0
    failed = []

    if args.individual:
        print(f"Uploading {len(files)} files individually...\n")
        for path in tqdm(files, desc="Files", unit="file"):
            try:
                with volume.batch_upload(force=args.overwrite) as batch:
                    batch.put_file(str(path), f"/{path.name}")
                uploaded += 1
            except Exception as e:
                failed.append((path.name, str(e)))
    else:
        print(f"Uploading {len(files)} files in batches of 5...\n")
        batch_size = 5
        for i in tqdm(range(0, len(files), batch_size), desc="Batches", unit="batch"):
            batch_files = files[i : i + batch_size]
            try:
                with volume.batch_upload(force=args.overwrite) as batch:
                    for path in batch_files:
                        batch.put_file(str(path), f"/{path.name}")
                uploaded += len(batch_files)
            except Exception as e:
                # Batch failed; check which files actually made it to volume
                try:
                    existing = set(volume.listdir("/"))
                except Exception:
                    existing = set()

                for path in batch_files:
                    if path.name in existing:
                        uploaded += 1
                    else:
                        failed.append((path.name, str(e)))

    print(f"\n{'='*60}")
    print(f"Upload complete: {uploaded}/{len(files)} files succeeded")
    if failed:
        print(f"\n⚠️  {len(failed)} files failed:")
        for fname, error in failed:
            print(f"  - {fname}: {error}")
    else:
        print("All files uploaded successfully!")
    print(f"{'='*60}")
    print("You can now run: modal run scripts/train_modal.py")


if __name__ == "__main__":
    main()
