"""Simple test: add and delete a file from Modal volume."""
import modal
import os

volume = modal.Volume.from_name("gram-data", create_if_missing=True)
app = modal.App("test-volume")


@app.function(volumes={"/data": volume})
def test_add_and_delete():
    """Add a file, verify it exists, delete it, verify deletion."""
    test_file = "/data/test_dummy.txt"

    # Test 1: Add file
    with open(test_file, "w") as f:
        f.write("test content")
    assert os.path.exists(test_file), "Failed to add file"
    print("✅ Added file")

    # Test 2: Delete file
    os.remove(test_file)
    assert not os.path.exists(test_file), "Failed to delete file"
    print("✅ Deleted file")


@app.local_entrypoint()
def main():
    test_add_and_delete.remote()
