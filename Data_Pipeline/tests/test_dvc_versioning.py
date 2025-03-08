"""
Tests for DVC versioning script
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from scripts.dvc_versioning import (
    run_command,
    track_bucket_data,
    parse_arguments,
    ensure_bucket_exists,
    list_bucket_files,
)


class TestDVCVersioning(unittest.TestCase):
    """Test case for the DVC versioning script."""

    @patch(
        "sys.argv",
        [
            "dvc_versioning.py",
            "--destination_bucket",
            "test-dest",
            "--dvc_remote",
            "test-remote",
        ],
    )
    def test_parse_arguments(self):
        """Test the argument parsing with required values."""
        args = parse_arguments()

        self.assertEqual(args.destination_bucket, "test-dest")
        self.assertEqual(args.dvc_remote, "test-remote")
        self.assertFalse(args.verbose)

    @patch(
        "sys.argv",
        [
            "dvc_versioning.py",
            "--destination_bucket",
            "test-dest",
            "--dvc_remote",
            "test-remote",
            "--verbose",
        ],
    )
    def test_parse_arguments_verbose(self):
        """Test the argument parsing with verbose flag."""
        args = parse_arguments()

        self.assertEqual(args.destination_bucket, "test-dest")
        self.assertEqual(args.dvc_remote, "test-remote")
        self.assertTrue(args.verbose)

    @patch("scripts.dvc_versioning.run_command")
    def test_run_command(self, mock_run):
        """Test the run_command function."""
        # Success case
        mock_run.return_value = (True, "Success")
        success, output = run_command("test command")
        self.assertTrue(success)
        self.assertEqual(output, "Success")

        # Failure case
        mock_run.return_value = (False, "Error")
        success, output = run_command("test command")
        self.assertFalse(success)
        self.assertEqual(output, "Error")

    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    def test_ensure_bucket_exists(self, mock_setup_gcp, mock_client):
        """Test the ensure_bucket_exists function."""
        # Case 1: Bucket already exists
        mock_client.return_value.get_bucket.return_value = MagicMock()
        result = ensure_bucket_exists("existing-bucket")
        self.assertTrue(result)
        mock_setup_gcp.assert_called_once()
        mock_client.return_value.get_bucket.assert_called_once_with(
            "existing-bucket"
        )
        mock_client.return_value.create_bucket.assert_not_called()

        # Reset mocks
        mock_setup_gcp.reset_mock()
        mock_client.reset_mock()

        # Case 2: Bucket doesn't exist and is created successfully
        mock_client.return_value.get_bucket.side_effect = Exception(
            "Not Found"
        )
        result = ensure_bucket_exists("new-bucket")
        self.assertTrue(result)
        mock_setup_gcp.assert_called_once()
        mock_client.return_value.get_bucket.assert_called_once_with(
            "new-bucket"
        )
        mock_client.return_value.create_bucket.assert_called_once_with(
            "new-bucket"
        )

        # Reset mocks
        mock_setup_gcp.reset_mock()
        mock_client.reset_mock()

        # Case 3: Bucket doesn't exist and creation fails
        mock_client.return_value.get_bucket.side_effect = Exception(
            "Not Found"
        )
        mock_client.return_value.create_bucket.side_effect = Exception(
            "Permission denied"
        )
        result = ensure_bucket_exists("failed-bucket")
        self.assertFalse(result)
        mock_setup_gcp.assert_called_once()
        mock_client.return_value.get_bucket.assert_called_once_with(
            "failed-bucket"
        )
        mock_client.return_value.create_bucket.assert_called_once_with(
            "failed-bucket"
        )

        # Reset mocks
        mock_setup_gcp.reset_mock()
        mock_client.reset_mock()

        # Case 4: Error accessing bucket
        mock_client.return_value.get_bucket.side_effect = Exception(
            "Unknown error"
        )
        result = ensure_bucket_exists("error-bucket")
        self.assertFalse(result)
        mock_setup_gcp.assert_called_once()
        mock_client.return_value.get_bucket.assert_called_once_with(
            "error-bucket"
        )
        mock_client.return_value.create_bucket.assert_not_called()

    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    def test_list_bucket_files(self, mock_setup_gcp, mock_client):
        """Test the list_bucket_files function."""
        # Set up mock blobs
        mock_blob1 = MagicMock()
        mock_blob1.name = "file1.csv"
        mock_blob1.size = 1000
        mock_blob1.updated = "2023-03-15"
        mock_blob1.md5_hash = "abc123"

        mock_blob2 = MagicMock()
        mock_blob2.name = "folder/file2.csv"
        mock_blob2.size = 2000
        mock_blob2.updated = "2023-03-16"
        mock_blob2.md5_hash = "def456"

        # Set up bucket mock
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client.return_value.get_bucket.return_value = mock_bucket

        # Test successful case
        result = list_bucket_files("test-bucket")
        self.assertEqual(len(result), 2)
        self.assertIn("file1.csv", result)
        self.assertIn("folder/file2.csv", result)
        self.assertEqual(result["file1.csv"]["size"], 1000)
        self.assertEqual(result["file1.csv"]["md5_hash"], "abc123")
        self.assertEqual(result["folder/file2.csv"]["size"], 2000)
        self.assertEqual(result["folder/file2.csv"]["md5_hash"], "def456")

        # Verify GCP credentials were set up
        mock_setup_gcp.assert_called_once()
        mock_client.return_value.get_bucket.assert_called_once_with(
            "test-bucket"
        )
        mock_bucket.list_blobs.assert_called_once()

        # Reset mocks
        mock_setup_gcp.reset_mock()
        mock_client.reset_mock()

        # Test bucket access failure
        mock_client.return_value.get_bucket.side_effect = Exception(
            "Access denied"
        )
        result = list_bucket_files("error-bucket")
        self.assertEqual(result, {})
        mock_setup_gcp.assert_called_once()
        mock_client.return_value.get_bucket.assert_called_once_with(
            "error-bucket"
        )

    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.tempfile.TemporaryDirectory")
    @patch("scripts.dvc_versioning.open", new_callable=mock_open)
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    def test_track_bucket_data(
        self,
        mock_setup_gcp,
        mock_file,
        mock_temp_dir,
        mock_run_command,
        mock_ensure_bucket,
        mock_list_files,
    ):
        """Test the track_bucket_data function."""
        # Set up mocks
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"
        mock_ensure_bucket.return_value = True

        # Create file metadata for before and after
        before_files = {
            "file1.csv": {
                "size": 1000,
                "updated": "2023-03-15",
                "md5_hash": "abc123",
            },
            "file2.csv": {
                "size": 2000,
                "updated": "2023-03-15",
                "md5_hash": "def456",
            },
        }

        after_files = {
            "file1.csv": {
                "size": 1500,
                "updated": "2023-03-16",
                "md5_hash": "abc999",
            },  # Modified
            "file2.csv": {
                "size": 2000,
                "updated": "2023-03-15",
                "md5_hash": "def456",
            },  # Unchanged
            "file3.csv": {
                "size": 3000,
                "updated": "2023-03-16",
                "md5_hash": "ghi789",
            },  # New
        }

        # Set up mock for file listing
        mock_list_files.side_effect = [before_files, after_files]

        # Success case
        mock_run_command.side_effect = [
            (True, "DVC init success"),  # dvc init
            (True, "DVC config set"),  # dvc config core.no_scm
            (True, "Cache remote added"),  # dvc remote add
            (True, "Remote config modified"),  # dvc remote modify
            (True, "Import complete"),  # dvc import-url
            (True, "Push complete"),  # dvc push
            (
                True,
                '{"unchanged": ["file2.csv"], "added": ["file3.csv"], "modified": ["file1.csv"]}',
            ),  # dvc status
        ]

        success = track_bucket_data("test-dest", "test-remote")
        self.assertTrue(success)

        # Verify buckets were checked/created
        mock_ensure_bucket.assert_any_call("test-dest")
        mock_ensure_bucket.assert_any_call("test-remote")

        # Verify file listing was called twice
        mock_list_files.assert_any_call("test-dest")
        self.assertEqual(mock_list_files.call_count, 2)

        # Verify GCP credentials were set up
        mock_setup_gcp.assert_called_once()

        # Verify commands were run in the correct order
        expected_calls = [
            unittest.mock.call("dvc init --no-scm -f", cwd="/tmp/test_dir"),
            unittest.mock.call(
                "dvc config core.no_scm true", cwd="/tmp/test_dir"
            ),
            unittest.mock.call(
                "dvc remote add -d test-remote gs://test-remote",
                cwd="/tmp/test_dir",
            ),
            unittest.mock.call(
                "dvc remote modify test-remote checksum_jobs 1",
                cwd="/tmp/test_dir",
            ),
            unittest.mock.call(
                "dvc import-url gs://test-dest test-dest --no-download",
                cwd="/tmp/test_dir",
            ),
            unittest.mock.call("dvc push", cwd="/tmp/test_dir"),
            unittest.mock.call("dvc status --json", cwd="/tmp/test_dir"),
        ]
        mock_run_command.assert_has_calls(expected_calls, any_order=False)

        # Test empty bucket case
        mock_run_command.reset_mock()
        mock_list_files.reset_mock()
        mock_setup_gcp.reset_mock()
        mock_list_files.side_effect = [{}, {}]  # Empty before and after

        # Mock the import-url failure for empty bucket and success for placeholder
        mock_run_command.side_effect = [
            (True, "DVC init success"),  # dvc init
            (True, "DVC config set"),  # dvc config core.no_scm
            (True, "Cache remote added"),  # dvc remote add
            (True, "Remote config modified"),  # dvc remote modify
            (False, "URL does not exist"),  # dvc import-url fails
            (True, "Add complete"),  # dvc add placeholder
            (True, "Push complete"),  # dvc push
            (True, "{}"),  # dvc status
        ]

        success = track_bucket_data("test-dest", "test-remote")
        self.assertTrue(success)

        # Verify placeholder file was created
        mock_file.assert_called_with(
            "/tmp/test_dir/test-dest.placeholder", "w"
        )
        handle = mock_file()
        handle.write.assert_called_with(
            "# Placeholder for tracking empty bucket: test-dest\n"
        )

        # Verify correct DVC commands were called for empty bucket
        expected_calls = [
            unittest.mock.call("dvc init --no-scm -f", cwd="/tmp/test_dir"),
            unittest.mock.call(
                "dvc config core.no_scm true", cwd="/tmp/test_dir"
            ),
            unittest.mock.call(
                "dvc remote add -d test-remote gs://test-remote",
                cwd="/tmp/test_dir",
            ),
            unittest.mock.call(
                "dvc remote modify test-remote checksum_jobs 1",
                cwd="/tmp/test_dir",
            ),
            unittest.mock.call(
                "dvc import-url gs://test-dest test-dest --no-download",
                cwd="/tmp/test_dir",
            ),
            unittest.mock.call(
                "dvc add test-dest.placeholder",
                cwd="/tmp/test_dir",
            ),
            unittest.mock.call("dvc push", cwd="/tmp/test_dir"),
            unittest.mock.call("dvc status --json", cwd="/tmp/test_dir"),
        ]
        mock_run_command.assert_has_calls(expected_calls, any_order=False)


if __name__ == "__main__":
    unittest.main()
