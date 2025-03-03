#!/usr/bin/env python3
"""Debug script to check GCS bucket contents from inside Docker"""

import os
import subprocess
import json

# Buckets to check
CACHE_BUCKET = "fully-processed-cache"
DVC_REMOTE = "fully-processed-data-dvc"

# Path to save results
OUTPUT_FILE = "/app/debug_results.json"

# Collect debug information
debug_info = {"environment": {}, "buckets": {}}


# Gather environment information
def collect_environment_info():
    debug_info["environment"]["creds_path"] = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    debug_info["environment"]["cwd"] = os.getcwd()

    # Check credential file
    creds_file = "/app/secret/gcp-key.json"
    debug_info["environment"]["creds_exists"] = os.path.exists(creds_file)
    if debug_info["environment"]["creds_exists"]:
        debug_info["environment"]["creds_size"] = os.path.getsize(creds_file)


# Check bucket files using gsutil directly
def check_bucket_with_gsutil(bucket_name):
    cmd = f"gsutil ls gs://{bucket_name}"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )
        files = []
        if result.returncode == 0 and result.stdout.strip():
            files = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]

        debug_info["buckets"][bucket_name] = {
            "gsutil_files": files,
            "gsutil_count": len(files),
            "gsutil_error": result.stderr if result.returncode != 0 else None,
        }
    except Exception as e:
        debug_info["buckets"][bucket_name] = {"gsutil_error": str(e)}


# Check bucket with Python
def check_bucket_with_python(bucket_name):
    try:
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs())

        files = []
        for blob in blobs:
            files.append(
                {
                    "name": blob.name,
                    "size": blob.size,
                    "updated": str(blob.updated),
                    "md5": blob.md5_hash,
                }
            )

        debug_info["buckets"][bucket_name]["python_files"] = files
        debug_info["buckets"][bucket_name]["python_count"] = len(files)
    except Exception as e:
        debug_info["buckets"][bucket_name]["python_error"] = str(e)


def main():
    print(f"Starting bucket debug in Docker container")

    collect_environment_info()
    print(f"Collected environment info")

    # Check cache bucket
    check_bucket_with_gsutil(CACHE_BUCKET)
    check_bucket_with_python(CACHE_BUCKET)
    print(f"Checked cache bucket: {CACHE_BUCKET}")

    # Check DVC remote bucket
    check_bucket_with_gsutil(DVC_REMOTE)
    check_bucket_with_python(DVC_REMOTE)
    print(f"Checked DVC remote bucket: {DVC_REMOTE}")

    # Save results to file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(debug_info, f, indent=2)

    print(f"Debug results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
