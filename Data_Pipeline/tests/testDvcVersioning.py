import unittest
from unittest.mock import MagicMock, patch, call
from argparse import Namespace
from subprocess import CalledProcessError
from scripts.dvc_versioning import *


class TestDVCVersioning(unittest.TestCase):


    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_bucket_already_exists(self, mock_storage_client, mock_logger):
        bucket_name = "existing-bucket"
        # Create a dummy bucket with exists() returning True.
        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = True

        # Setup the storage client mock.
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        # When create_bucket is called it should not be used in this test.
        mock_storage_client.return_value = mock_storage_instance

        # Call the function.
        result = ensure_bucket_exists(bucket_name)
        
        # Assert that the bucket already exists and True is returned.
        self.assertTrue(result)
        dummy_bucket.exists.assert_called_once()
        # Optionally, check that a log message was written.
        mock_logger.info.assert_any_call(f"Bucket {bucket_name} already exists")


    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_bucket_not_exists_and_creation_successful(self, mock_storage_client, mock_logger):
        bucket_name = "new-bucket"
        # Create a dummy bucket with exists() returning False.
        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = False

        # Setup the storage client mock.
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        # Simulate successful creation by returning the dummy bucket.
        mock_storage_instance.create_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call the function.
        result = ensure_bucket_exists(bucket_name)
        
        # Assert that bucket creation was attempted and True is returned.
        self.assertTrue(result)
        dummy_bucket.exists.assert_called_once()
        mock_storage_instance.create_bucket.assert_called_once_with(bucket_name)
        mock_logger.info.assert_any_call(f"Bucket {bucket_name} created successfully")



    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_exception_during_bucket_exists(self, mock_storage_client, mock_logger):
        bucket_name = "error-bucket"
        # Create a dummy bucket whose exists() method raises an Exception.
        dummy_bucket = MagicMock()
        dummy_bucket.exists.side_effect = Exception("exists() failure")

        # Setup the storage client mock.
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call the function and expect it to return False.
        result = ensure_bucket_exists(bucket_name)
        self.assertFalse(result)
        dummy_bucket.exists.assert_called_once()
        mock_logger.error.assert_any_call(
            f"Error checking/creating bucket {bucket_name}: exists() failure"
        )


    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_exception_during_create_bucket(self, mock_storage_client, mock_logger):
        bucket_name = "new-bucket"
        # Create a dummy bucket with exists() returning False.
        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = False

        # Setup the storage client mock.
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        # Simulate an exception when trying to create the bucket.
        mock_storage_instance.create_bucket.side_effect = Exception("create_bucket failure")
        mock_storage_client.return_value = mock_storage_instance

        # Call the function and expect it to return False.
        result = ensure_bucket_exists(bucket_name)
        self.assertFalse(result)
        dummy_bucket.exists.assert_called_once()
        mock_storage_instance.create_bucket.assert_called_once_with(bucket_name)
        mock_logger.error.assert_any_call(
            f"Error checking/creating bucket {bucket_name}: create_bucket failure"
        )



    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_bucket_does_not_exist(self, mock_storage_client, mock_logger):
        bucket_name = "nonexistent-bucket"

        # Simulate bucket.exists() returning False
        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = False

        # Setup the storage client to return our dummy bucket.
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call clear_bucket
        result = clear_bucket(bucket_name)

        # Verify that a warning was logged and True is returned.
        self.assertTrue(result)
        dummy_bucket.exists.assert_called_once()
        mock_logger.warning.assert_called_once_with(f"Bucket {bucket_name} does not exist")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_bucket_exists_no_blobs(self, mock_storage_client, mock_logger):
        bucket_name = "empty-bucket"

        # Simulate bucket.exists() returning True and list_blobs() returning an empty iterator.
        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = True
        dummy_bucket.list_blobs.return_value = iter([])

        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call clear_bucket
        result = clear_bucket(bucket_name)

        # Assert that the function returns True and proper log messages were recorded.
        self.assertTrue(result)
        dummy_bucket.exists.assert_called_once()
        dummy_bucket.list_blobs.assert_called_once()
        mock_logger.info.assert_any_call(f"Bucket {bucket_name} cleared successfully")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_bucket_exists_with_blobs(self, mock_storage_client, mock_logger):
        bucket_name = "bucket-with-blobs"

        # Create dummy blobs with a name and a delete method.
        dummy_blob1 = MagicMock()
        dummy_blob1.name = "blob1.txt"
        dummy_blob2 = MagicMock()
        dummy_blob2.name = "blob2.txt"

        # Simulate bucket.exists() returns True and list_blobs() returns our dummy blobs.
        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = True
        dummy_bucket.list_blobs.return_value = [dummy_blob1, dummy_blob2]

        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call clear_bucket
        result = clear_bucket(bucket_name)

        # Assert that delete was called for each blob and True is returned.
        self.assertTrue(result)
        dummy_bucket.exists.assert_called_once()
        dummy_bucket.list_blobs.assert_called_once()
        dummy_blob1.delete.assert_called_once()
        dummy_blob2.delete.assert_called_once()
        # Check that debug logs were written for each blob deletion.
        expected_debug_calls = [
            call(f"Deleting {dummy_blob1.name}"),
            call(f"Deleting {dummy_blob2.name}")
        ]
        mock_logger.debug.assert_has_calls(expected_debug_calls, any_order=True)
        mock_logger.info.assert_any_call(f"Bucket {bucket_name} cleared successfully")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_exception_during_list_blobs(self, mock_storage_client, mock_logger):
        bucket_name = "bucket-error-list"

        # Simulate bucket.exists() returns True and list_blobs() raises an Exception.
        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = True
        dummy_bucket.list_blobs.side_effect = Exception("list_blobs error")

        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call clear_bucket and expect it to return False.
        result = clear_bucket(bucket_name)
        self.assertFalse(result)
        dummy_bucket.exists.assert_called_once()
        dummy_bucket.list_blobs.assert_called_once()
        mock_logger.error.assert_called_once()
        # Optionally check that the error message contains the exception text.
        error_call_args = mock_logger.error.call_args[0][0]
        self.assertIn("list_blobs error", error_call_args)

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_exception_during_blob_deletion(self, mock_storage_client, mock_logger):
        bucket_name = "bucket-error-delete"

        # Create a dummy blob that raises an Exception on deletion.
        dummy_blob = MagicMock()
        dummy_blob.name = "error_blob.txt"
        dummy_blob.delete.side_effect = Exception("delete error")

        dummy_bucket = MagicMock()
        dummy_bucket.exists.return_value = True
        dummy_bucket.list_blobs.return_value = [dummy_blob]

        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call clear_bucket and expect it to return False.
        result = clear_bucket(bucket_name)
        self.assertFalse(result)
        dummy_bucket.exists.assert_called_once()
        dummy_bucket.list_blobs.assert_called_once()
        dummy_blob.delete.assert_called_once()
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        self.assertIn("delete error", error_call_args)


    
    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_list_bucket_files_with_files_and_directories(self, mock_storage_client, mock_logger):
        bucket_name = "test-bucket"
        
        # Create dummy blobs: two files and one directory.
        dummy_blob1 = MagicMock()
        dummy_blob1.name = "file1.txt"
        dummy_blob2 = MagicMock()
        dummy_blob2.name = "file2.txt"
        dummy_blob3 = MagicMock()
        dummy_blob3.name = "subdir/"
        
        dummy_blobs = [dummy_blob1, dummy_blob2, dummy_blob3]
        
        # Setup the storage client mock so that list_blobs returns our dummy blobs.
        mock_storage_instance = MagicMock()
        mock_storage_instance.list_blobs.return_value = dummy_blobs
        mock_storage_client.return_value = mock_storage_instance
        
        # Call the function.
        result = list_bucket_files(bucket_name)
        
        # Only file names (non-directory) should be returned.
        expected = ["file1.txt", "file2.txt"]
        self.assertEqual(result, expected)
        
        # Verify that the logger recorded the number of files found.
        mock_logger.info.assert_called_once_with(f"Found {len(expected)} files in bucket {bucket_name}")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_list_bucket_files_no_files(self, mock_storage_client, mock_logger):
        bucket_name = "empty-bucket"
        
        # Setup the storage client mock: list_blobs returns an empty list.
        mock_storage_instance = MagicMock()
        mock_storage_instance.list_blobs.return_value = []
        mock_storage_client.return_value = mock_storage_instance
        
        # Call the function.
        result = list_bucket_files(bucket_name)
        
        # Expect an empty list as the result.
        self.assertEqual(result, [])
        mock_logger.info.assert_called_once_with(f"Found 0 files in bucket {bucket_name}")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_list_bucket_files_exception(self, mock_storage_client, mock_logger):
        bucket_name = "error-bucket"
        
        # Simulate an exception when list_blobs is called.
        mock_storage_instance = MagicMock()
        mock_storage_instance.list_blobs.side_effect = Exception("Test exception")
        mock_storage_client.return_value = mock_storage_instance
        
        # Call the function and expect an empty list.
        result = list_bucket_files(bucket_name)
        self.assertEqual(result, [])
        
        # Verify that an error was logged and contains the exception message.
        self.assertTrue(
            any("Test exception" in str(arg) for args, _ in mock_logger.error.call_args_list for arg in args),
            "Expected logger.error to be called with the exception message."
        )


    
    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_less_than_20_files(self, mock_storage_client, mock_logger):
        dvc_remote = "test-remote"
        # Create dummy blobs with names.
        dummy_blobs = [MagicMock(name=f"blob{i}") for i in range(3)]
        for i, blob in enumerate(dummy_blobs):
            blob.name = f"file{i}.txt"

        # Setup the storage client mock.
        mock_storage_instance = MagicMock()
        mock_storage_instance.list_blobs.return_value = dummy_blobs
        mock_storage_client.return_value = mock_storage_instance

        # Call the function.
        result = list_dvc_remote_contents(dvc_remote)

        # Expected file names.
        expected_files = [f"file{i}.txt" for i in range(3)]
        self.assertEqual(result, expected_files)

        # Verify header log call.
        mock_logger.info.assert_any_call("DVC Remote Contents (3 files):")
        expected_file_calls = [call(f"  - file{i}.txt") for i in range(3)]
        for expected_call in expected_file_calls:
            self.assertIn(expected_call, mock_logger.info.call_args_list)
        for log_call in mock_logger.info.call_args_list:
            self.assertNotIn("... and", log_call[0][0])

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_exactly_20_files(self, mock_storage_client, mock_logger):
        dvc_remote = "test-remote"
        # Create exactly 20 dummy blobs.
        dummy_blobs = [MagicMock(name=f"blob{i}") for i in range(20)]
        for i, blob in enumerate(dummy_blobs):
            blob.name = f"file{i}.txt"

        mock_storage_instance = MagicMock()
        mock_storage_instance.list_blobs.return_value = dummy_blobs
        mock_storage_client.return_value = mock_storage_instance

        result = list_dvc_remote_contents(dvc_remote)
        expected_files = [f"file{i}.txt" for i in range(20)]
        self.assertEqual(result, expected_files)

        # Verify header logging.
        mock_logger.info.assert_any_call("DVC Remote Contents (20 files):")
        # Verify that each of the 20 file names is logged.
        expected_file_calls = [call(f"  - file{i}.txt") for i in range(20)]
        for expected_call in expected_file_calls:
            self.assertIn(expected_call, mock_logger.info.call_args_list)
        # There should be no extra "more files" log.
        for log_call in mock_logger.info.call_args_list:
            if "DVC Remote Contents" not in log_call[0][0] and "  - " not in log_call[0][0]:
                self.assertNotIn("... and", log_call[0][0])

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_more_than_20_files(self, mock_storage_client, mock_logger):
        dvc_remote = "test-remote"
        total_files = 25
        # Create 25 dummy blobs.
        dummy_blobs = [MagicMock(name=f"blob{i}") for i in range(total_files)]
        for i, blob in enumerate(dummy_blobs):
            blob.name = f"file{i}.txt"

        mock_storage_instance = MagicMock()
        mock_storage_instance.list_blobs.return_value = dummy_blobs
        mock_storage_client.return_value = mock_storage_instance

        result = list_dvc_remote_contents(dvc_remote)
        expected_files = [f"file{i}.txt" for i in range(total_files)]
        self.assertEqual(result, expected_files)

        # Verify header log call.
        mock_logger.info.assert_any_call("DVC Remote Contents (25 files):")
        # Verify that the first 20 files are logged.
        expected_file_calls = [call(f"  - file{i}.txt") for i in range(20)]
        for expected_call in expected_file_calls:
            self.assertIn(expected_call, mock_logger.info.call_args_list)
        # Verify that the extra message is logged.
        mock_logger.info.assert_any_call("  ... and 5 more files")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.storage.Client")
    def test_exception_in_list_blobs(self, mock_storage_client, mock_logger):
        dvc_remote = "test-remote"
        # Simulate an exception when calling list_blobs.
        mock_storage_instance = MagicMock()
        mock_storage_instance.list_blobs.side_effect = Exception("list_blobs failure")
        mock_storage_client.return_value = mock_storage_instance

        result = list_dvc_remote_contents(dvc_remote)
        self.assertEqual(result, [])
        # Verify that an error was logged that includes the exception message.
        self.assertTrue(
            any("list_blobs failure" in str(arg) for args, _ in mock_logger.error.call_args_list for arg in args)
        )


    
    @patch("scripts.dvc_versioning.datetime")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.logger")
    def test_save_version_metadata_success_existing(self, mock_logger, mock_storage_client, mock_datetime):
        dvc_remote = "dvc-remote-bucket"
        cache_bucket = "cache-bucket"
        file_info = [{"file": "a.txt"}]
        metadata_blob_name = f"dvc_metadata/{cache_bucket}_versions.json"
        
        # Set a fixed datetime.
        fixed_datetime = datetime(2025, 3, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_datetime

        # Create a dummy blob that returns existing metadata.
        existing_metadata = {"versions": [{"timestamp": "2025-01-01T00:00:00", "files": []}]}
        dummy_blob = MagicMock()
        dummy_blob.download_as_text.return_value = json.dumps(existing_metadata)
        
        # Set up the dummy bucket to return the blob.
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob

        # Configure storage client mock.
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance

        # Call the function.
        result = save_version_metadata(dvc_remote, cache_bucket, file_info)
        
        # Expect success.
        self.assertTrue(result)
        # The new version should use the fixed timestamp.
        new_version = {"timestamp": fixed_datetime.isoformat(), "files": file_info}
        expected_metadata = {"versions": existing_metadata["versions"] + [new_version]}
        expected_json = json.dumps(expected_metadata, indent=2)
        
        dummy_blob.upload_from_string.assert_called_once_with(expected_json)
        mock_logger.info.assert_any_call(f"Saved version metadata to {metadata_blob_name}")

    @patch("scripts.dvc_versioning.datetime")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.logger")
    def test_save_version_metadata_success_not_found(self, mock_logger, mock_storage_client, mock_datetime):
        dvc_remote = "dvc-remote-bucket"
        cache_bucket = "cache-bucket"
        file_info = [{"file": "b.txt"}]
        metadata_blob_name = f"dvc_metadata/{cache_bucket}_versions.json"
        
        fixed_datetime = datetime(2025, 3, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_datetime
        
        # Simulate NotFound when trying to download metadata.
        dummy_blob = MagicMock()
        dummy_blob.download_as_text.side_effect = NotFound("Not Found")
        
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance
        
        result = save_version_metadata(dvc_remote, cache_bucket, file_info)
        
        self.assertTrue(result)
        new_version = {"timestamp": fixed_datetime.isoformat(), "files": file_info}
        expected_metadata = {"versions": [new_version]}
        expected_json = json.dumps(expected_metadata, indent=2)
        
        dummy_blob.upload_from_string.assert_called_once_with(expected_json)
        mock_logger.info.assert_any_call(f"Saved version metadata to {metadata_blob_name}")

    @patch("scripts.dvc_versioning.datetime")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.logger")
    def test_save_version_metadata_success_generic_exception_on_download(self, mock_logger, mock_storage_client, mock_datetime):
        dvc_remote = "dvc-remote-bucket"
        cache_bucket = "cache-bucket"
        file_info = [{"file": "c.txt"}]
        metadata_blob_name = f"dvc_metadata/{cache_bucket}_versions.json"
        
        fixed_datetime = datetime(2025, 3, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_datetime
        
        # Simulate a generic exception during download.
        dummy_blob = MagicMock()
        dummy_blob.download_as_text.side_effect = Exception("Download error")
        
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance
        
        result = save_version_metadata(dvc_remote, cache_bucket, file_info)
        
        self.assertTrue(result)
        new_version = {"timestamp": fixed_datetime.isoformat(), "files": file_info}
        expected_metadata = {"versions": [new_version]}
        expected_json = json.dumps(expected_metadata, indent=2)
        
        dummy_blob.upload_from_string.assert_called_once_with(expected_json)
        mock_logger.info.assert_any_call(f"Saved version metadata to {metadata_blob_name}")

    @patch("scripts.dvc_versioning.datetime")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.logger")
    def test_save_version_metadata_failure_on_upload(self, mock_logger, mock_storage_client, mock_datetime):
        dvc_remote = "dvc-remote-bucket"
        cache_bucket = "cache-bucket"
        file_info = [{"file": "d.txt"}]
        
        fixed_datetime = datetime(2025, 3, 15, 12, 0, 0)
        mock_datetime.now.return_value = fixed_datetime
        
        dummy_blob = MagicMock()
        # Simulate valid metadata download.
        existing_metadata = {"versions": []}
        dummy_blob.download_as_text.return_value = json.dumps(existing_metadata)
        # Simulate an exception during upload.
        dummy_blob.upload_from_string.side_effect = Exception("Upload error")
        
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        
        mock_storage_instance = MagicMock()
        mock_storage_instance.bucket.return_value = dummy_bucket
        mock_storage_client.return_value = mock_storage_instance
        
        result = save_version_metadata(dvc_remote, cache_bucket, file_info)
        
        self.assertFalse(result)
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        self.assertIn("Upload error", error_message)


    def test_required_arguments_only(self):
        test_args = [
            "prog",
            "--cache_bucket", "my-cache",
            "--dvc_remote", "my-dvc"
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_arguments()
        self.assertEqual(args.cache_bucket, "my-cache")
        self.assertEqual(args.dvc_remote, "my-dvc")
        # Boolean flags default to False
        self.assertFalse(args.clear_remote)
        self.assertFalse(args.debug)
        self.assertFalse(args.keep_temp)
        # Default value for gcp_key_path is used
        self.assertEqual(args.gcp_key_path, "/app/secret/gcp-key.json")

    def test_all_arguments(self):
        test_args = [
            "prog",
            "--cache_bucket", "cache",
            "--dvc_remote", "dvc",
            "--clear_remote",
            "--debug",
            "--keep_temp",
            "--gcp_key_path", "path/to/key.json"
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_arguments()
        self.assertEqual(args.cache_bucket, "cache")
        self.assertEqual(args.dvc_remote, "dvc")
        self.assertTrue(args.clear_remote)
        self.assertTrue(args.debug)
        self.assertTrue(args.keep_temp)
        self.assertEqual(args.gcp_key_path, "path/to/key.json")

    def test_missing_required_arguments(self):
        # Omitting the required --dvc_remote should cause a SystemExit.
        test_args = ["prog", "--cache_bucket", "cache"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                parse_arguments()


    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.subprocess.run")
    def test_run_command_success(self, mock_run, mock_logger):
        # Simulate a successful command execution.
        fake_result = subprocess.CompletedProcess(
            args="echo hello",
            returncode=0,
            stdout="hello\n"
        )
        mock_run.return_value = fake_result

        success, output = run_command("echo hello")
        self.assertTrue(success)
        self.assertEqual(output, "hello")
        mock_logger.info.assert_any_call("Command completed successfully")
        mock_logger.debug.assert_any_call("Output: hello")

        # Verify that the environment passed to subprocess.run includes PYTHONUNBUFFERED.
        _, kwargs = mock_run.call_args
        env = kwargs.get("env", {})
        self.assertEqual(env.get("PYTHONUNBUFFERED"), "1")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.subprocess.run")
    def test_run_command_failure_return(self, mock_run, mock_logger):
        # Simulate a command that fails but does not raise an exception (check=False).
        fake_result = subprocess.CompletedProcess(
            args="false",
            returncode=1,
            stdout="error occurred\n"
        )
        mock_run.return_value = fake_result

        success, output = run_command("false", check=False)
        self.assertFalse(success)
        self.assertEqual(output, "error occurred")
        mock_logger.error.assert_any_call("Command failed with exit code 1")
        mock_logger.error.assert_any_call("Error output: error occurred")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.subprocess.run")
    def test_run_command_called_process_error(self, mock_run, mock_logger):
        # Simulate subprocess.run raising a CalledProcessError.
        error = CalledProcessError(returncode=2, cmd="bad command", output="bad output")
        mock_run.side_effect = error

        success, output = run_command("bad command", check=True)
        self.assertFalse(success)
        self.assertEqual(output, str(error))
        mock_logger.error.assert_any_call(f"Command execution failed: {error}")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.subprocess.run")
    def test_run_command_general_exception(self, mock_run, mock_logger):
        # Simulate a generic exception raised by subprocess.run.
        mock_run.side_effect = Exception("general error")

        success, output = run_command("some command")
        self.assertFalse(success)
        self.assertEqual(output, "general error")
        mock_logger.error.assert_any_call("Error running command: general error")

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.subprocess.run")
    def test_run_command_env_update(self, mock_run, mock_logger):
        # Verify that additional environment variables are passed along.
        fake_result = subprocess.CompletedProcess(
            args="echo env",
            returncode=0,
            stdout="env output\n"
        )
        mock_run.return_value = fake_result

        custom_env = {"MY_VAR": "value"}
        success, output = run_command("echo env", env=custom_env)
        self.assertTrue(success)
        self.assertEqual(output, "env output")

        # Check that the environment passed to subprocess.run includes both MY_VAR and PYTHONUNBUFFERED.
        _, kwargs = mock_run.call_args
        env = kwargs.get("env", {})
        self.assertEqual(env.get("MY_VAR"), "value")
        self.assertEqual(env.get("PYTHONUNBUFFERED"), "1")



    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_dvc_init_failure(self, mock_logger, mock_run_command):
        # Simulate failure on the first command: "dvc init --no-scm"
        mock_run_command.return_value = (False, "init error")
        temp_dir = "/tmp/test"
        dvc_remote = "my-dvc-bucket"

        result = setup_and_verify_dvc_remote(temp_dir, dvc_remote)
        self.assertFalse(result)
        mock_run_command.assert_called_once_with("dvc init --no-scm", cwd=temp_dir)
        mock_logger.error.assert_any_call("Failed to initialize DVC: init error")

    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_dvc_remote_add_failure(self, mock_logger, mock_run_command):
        # First call (init) succeeds, second (remote add) fails.
        # Configure side_effect for sequential calls.
        # Call 1: init, Call 2: remote add.
        mock_run_command.side_effect = [
            (True, "init ok"),
            (False, "remote add error")
        ]
        temp_dir = "/tmp/test"
        dvc_remote = "my-dvc-bucket"

        result = setup_and_verify_dvc_remote(temp_dir, dvc_remote)
        self.assertFalse(result)
        # Verify that "dvc init" and "dvc remote add" were called.
        expected_init_cmd = "dvc init --no-scm"
        expected_remote_cmd = f"dvc remote add -d myremote gs://{dvc_remote}"
        calls = [
            call(expected_init_cmd, cwd=temp_dir),
            call(expected_remote_cmd, cwd=temp_dir)
        ]
        mock_run_command.assert_has_calls(calls)
        mock_logger.error.assert_any_call("Failed to add DVC remote: remote add error")

    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_successful_without_gcp_and_debug(self, mock_logger, mock_run_command):
        # Simulate a successful flow without providing a gcp_key and with debug_mode off.
        # Expected run_command calls:
        # 1. dvc init
        # 2. dvc remote add
        # 3. dvc remote modify myremote checksum_jobs 16
        # 4. dvc remote modify myremote jobs 4
        # 5. Test remote connection command
        mock_run_command.side_effect = [
            (True, "init ok"),                   
            (True, "remote add ok"),             
            (True, "modify checksum ok"),        
            (True, "modify jobs ok"),            
            (True, "push ok")                    
        ]
        temp_dir = "/tmp/test"
        dvc_remote = "my-dvc-bucket"

        result = setup_and_verify_dvc_remote(temp_dir, dvc_remote)
        self.assertTrue(result)
        # There should be exactly 5 calls.
        self.assertEqual(mock_run_command.call_count, 5)

    @patch("scripts.dvc_versioning.os.path.exists")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_successful_with_gcp(self, mock_logger, mock_run_command, mock_path_exists):
        # Test with gcp_key_path provided and the file exists.
        # Expected calls: dvc init, remote add, two modify commands, credential modify, then remote test.
        mock_run_command.side_effect = [
            (True, "init ok"),                   
            (True, "remote add ok"),             
            (True, "modify checksum ok"),        
            (True, "modify jobs ok"),            
            (True, "credential modify ok"),      
            (True, "push ok")                    
        ]
        # Simulate that the provided gcp_key_path exists.
        mock_path_exists.return_value = True

        temp_dir = "/tmp/test"
        dvc_remote = "my-dvc-bucket"
        gcp_key_path = "/fake/path/key.json"

        result = setup_and_verify_dvc_remote(temp_dir, dvc_remote, gcp_key_path=gcp_key_path)
        self.assertTrue(result)
        # Expect 6 calls (the extra one for credentials).
        self.assertEqual(mock_run_command.call_count, 6)
        # Verify that the credentials command was called.
        expected_credential_cmd = f"dvc remote modify --local myremote credentialpath {gcp_key_path}"
        mock_run_command.assert_any_call(expected_credential_cmd, cwd=temp_dir)

    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_successful_with_debug_mode(self, mock_logger, mock_run_command):
        # Simulate a successful flow with debug mode enabled (no gcp key provided).
        # Expected calls:
        # 1. dvc init
        # 2. dvc remote add
        # 3. modify checksum_jobs
        # 4. modify jobs
        # 5. debug: cat .dvc/config
        # 6. debug: cat .dvc/config.local
        # 7. test remote connection
        # 8. debug: find .dvc -type f | sort
        mock_run_command.side_effect = [
            (True, "init ok"),                   
            (True, "remote add ok"),             
            (True, "modify checksum ok"),        
            (True, "modify jobs ok"),            
            (True, "config content"),            
            (True, "config.local content"),      
            (True, "push ok"),                   
            (True, "find output")                
        ]
        temp_dir = "/tmp/test"
        dvc_remote = "my-dvc-bucket"
        debug_mode = True

        result = setup_and_verify_dvc_remote(temp_dir, dvc_remote, debug_mode=debug_mode)
        self.assertTrue(result)
        # Expect 8 calls in total.
        self.assertEqual(mock_run_command.call_count, 8)
        # Verify that debug logging for configuration is triggered.
        mock_logger.info.assert_any_call("DVC configuration:")

    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_final_remote_test_failure(self, mock_logger, mock_run_command):
        # Simulate a failure in the final test remote connection.
        mock_run_command.side_effect = [
            (True, "init ok"),                   # dvc init
            (True, "remote add ok"),             # remote add
            (True, "modify checksum ok"),        # modify checksum_jobs
            (True, "modify jobs ok"),            # modify jobs
            (False, "push failed")               # test remote connection fails
        ]
        temp_dir = "/tmp/test"
        dvc_remote = "my-dvc-bucket"

        result = setup_and_verify_dvc_remote(temp_dir, dvc_remote)
        self.assertFalse(result)
        # Verify that "Testing DVC remote connection" was logged.
        mock_logger.info.assert_any_call("Testing DVC remote connection")
        # Check that final failure message led to returning False.
        self.assertEqual(mock_run_command.call_count, 5)



    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_bucket_existence_failure(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Simulate that ensure_bucket_exists returns False for "cache-bucket".
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.side_effect = lambda bucket: False if bucket == "cache-bucket" else True

        result = track_bucket_data("cache-bucket", "dvc-remote")
        self.assertFalse(result)
        mock_logger.error.assert_any_call("Failed to ensure bucket cache-bucket exists")
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_clear_remote_failure(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Buckets exist but clear_remote is requested and clear_bucket returns False.
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.return_value = True
        mock_clear_bucket.return_value = False

        result = track_bucket_data("cache-bucket", "dvc-remote", clear_remote=True)
        self.assertFalse(result)
        mock_logger.error.assert_any_call("Failed to clear bucket dvc-remote")
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_empty_bucket(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Buckets exist, but the cache bucket has no files.
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.return_value = True
        mock_list_bucket_files.return_value = []

        result = track_bucket_data("cache-bucket", "dvc-remote")
        self.assertTrue(result)
        mock_logger.warning.assert_any_call("No files found in bucket cache-bucket")
        mock_save_version_metadata.assert_called_once_with("dvc-remote", "cache-bucket", [])
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_setup_and_verify_failure(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Buckets exist and files are present, but setting up DVC remote fails.
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.return_value = True
        mock_list_bucket_files.return_value = ["file1.txt"]
        mock_setup_and_verify.return_value = False

        result = track_bucket_data("cache-bucket", "dvc-remote")
        self.assertFalse(result)
        mock_logger.error.assert_any_call("Failed to set up and verify DVC remote")
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_successful_processing_all_files(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Simulate a successful processing of two files.
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.return_value = True
        mock_clear_bucket.return_value = True  # not used since clear_remote is False
        mock_list_bucket_files.return_value = ["file1.txt", "file2.txt"]
        mock_setup_and_verify.return_value = True

        # run_command side effects for initial DVC commands and per-file processing:
        # 1. dvc init
        # 2. dvc remote add
        # 3. modify checksum_jobs
        # 4. modify jobs
        # 5. For file1.txt: dvc add
        # 6. For file1.txt: dvc push
        # 7. For file2.txt: dvc add
        # 8. For file2.txt: dvc push
        run_cmd_side_effects = [
            (True, "init ok"),
            (True, "remote add ok"),
            (True, "modify checksum ok"),
            (True, "modify jobs ok"),
            (True, "dvc add ok"),
            (True, "dvc push ok"),
            (True, "dvc add ok"),
            (True, "dvc push ok"),
        ]
        mock_run_command.side_effect = run_cmd_side_effects

        # Set up storage.Client to simulate blob operations.
        dummy_blob = MagicMock()
        dummy_blob.size = 123
        dummy_blob.md5_hash = "abc123"
        dummy_blob.updated = datetime(2025, 3, 15, 12, 0, 0)
        dummy_blob.generation = "gen1"
        dummy_blob.reload.return_value = None
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        result = track_bucket_data("cache-bucket", "dvc-remote")
        self.assertTrue(result)
        expected_file_info = [
            {
                "file_path": "file1.txt",
                "size": 123,
                "md5": "abc123",
                "updated": dummy_blob.updated.isoformat(),
                "generation": "gen1",
            },
            {
                "file_path": "file2.txt",
                "size": 123,
                "md5": "abc123",
                "updated": dummy_blob.updated.isoformat(),
                "generation": "gen1",
            },
        ]
        mock_save_version_metadata.assert_called_once_with("dvc-remote", "cache-bucket", expected_file_info)
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)

    
    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_exception_in_track_bucket_data_outer_try(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Setup: Create a fake temporary directory.
        mock_mkdtemp.return_value = "/tmp/fake"
        # Force ensure_bucket_exists to raise an exception.
        mock_ensure_bucket_exists.side_effect = Exception("Test exception")
        
        # Call the function.
        result = track_bucket_data("cache-bucket", "dvc-remote")
        
        self.assertFalse(result)
        mock_logger.error.assert_any_call("Error in track_bucket_data: Test exception")
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)



    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_partial_failure_processing(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Two files are found. For file1.txt, dvc add fails.
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.return_value = True
        mock_list_bucket_files.return_value = ["file1.txt", "file2.txt"]
        mock_setup_and_verify.return_value = True


        run_cmd_side_effects = [
            (False, "dvc add error"),  # file1.txt: dvc add fails
            (True, "dvc add ok"),      # file2.txt: dvc add succeeds
            (True, "dvc push ok"),     # file2.txt: dvc push succeeds
        ]
        mock_run_command.side_effect = run_cmd_side_effects

        # Make sure each call to bucket.blob(file_path) returns a fresh blob.
        def blob_side_effect(file_path):
            blob = MagicMock()
            blob.size = 456
            blob.md5_hash = "def456"
            blob.updated = datetime(2025, 3, 16, 12, 0, 0)
            blob.generation = "gen2"
            blob.reload.return_value = None
            blob.download_to_filename.return_value = None
            return blob

        dummy_bucket = MagicMock()
        dummy_bucket.blob.side_effect = blob_side_effect

        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        result = track_bucket_data("cache-bucket", "dvc-remote")
        self.assertTrue(result)
        # Expected file info should only include file2.txt.
        expected_file_info = [
            {
                "file_path": "file2.txt",
                "size": 456,
                "md5": "def456",
                "updated": "2025-03-16T12:00:00",
                "generation": "gen2",
            }
        ]
        mock_save_version_metadata.assert_called_once_with("dvc-remote", "cache-bucket", expected_file_info)
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)



    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_exception_in_file_processing(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # Simulate an exception during processing (e.g. during file download).
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.return_value = True
        mock_list_bucket_files.return_value = ["file1.txt"]
        mock_setup_and_verify.return_value = True

        # Initial DVC setup commands succeed.
        run_cmd_side_effects = [
            (True, "init ok"),
            (True, "remote add ok"),
            (True, "modify checksum ok"),
            (True, "modify jobs ok"),
        ]
        mock_run_command.side_effect = run_cmd_side_effects

        # Configure storage.Client so that blob.download_to_filename raises an exception.
        dummy_blob = MagicMock()
        dummy_blob.reload.return_value = None
        dummy_blob.download_to_filename.side_effect = Exception("download error")
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        result = track_bucket_data("cache-bucket", "dvc-remote")
        self.assertFalse(result)
        mock_logger.error.assert_any_call("Error processing file file1.txt: download error")
        # Even though file processing failed, version metadata is still saved (with empty file_info).
        mock_save_version_metadata.assert_called_once_with("dvc-remote", "cache-bucket", [])
        mock_rmtree.assert_called_once_with("/tmp/fake", ignore_errors=True)

    @patch("scripts.dvc_versioning.logger")
    @patch("scripts.dvc_versioning.setup_gcp_credentials")
    @patch("scripts.dvc_versioning.ensure_bucket_exists")
    @patch("scripts.dvc_versioning.clear_bucket")
    @patch("scripts.dvc_versioning.list_bucket_files")
    @patch("scripts.dvc_versioning.save_version_metadata")
    @patch("scripts.dvc_versioning.setup_and_verify_dvc_remote")
    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.list_dvc_remote_contents")
    @patch("scripts.dvc_versioning.storage.Client")
    @patch("scripts.dvc_versioning.shutil.rmtree")
    @patch("scripts.dvc_versioning.tempfile.mkdtemp")
    def test_keep_temp_behavior(
        self,
        mock_mkdtemp,
        mock_rmtree,
        mock_storage_client,
        mock_list_dvc,
        mock_run_command,
        mock_setup_and_verify,
        mock_save_version_metadata,
        mock_list_bucket_files,
        mock_clear_bucket,
        mock_ensure_bucket_exists,
        mock_setup_gcp,
        mock_logger,
    ):
        # When keep_temp is True, the temporary directory should not be removed.
        mock_mkdtemp.return_value = "/tmp/fake"
        mock_ensure_bucket_exists.return_value = True
        mock_list_bucket_files.return_value = []
        result = track_bucket_data("cache-bucket", "dvc-remote", keep_temp=True)
        self.assertTrue(result)
        mock_rmtree.assert_not_called()
        mock_logger.info.assert_any_call("Keeping temporary directory for debugging: /tmp/fake")


    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_debug_dvc_setup_all_success(self, mock_logger, mock_run_command):
        # Set up side effects in the order of run_command calls:
        # 1. Check DVC config files: output ignored.
        # 2. "cat .dvc/config" returns (True, "config content")
        # 3. "cat .dvc/config.local" returns (True, "config.local content")
        # 4. "find . -name '*.dvc'" returns (True, "dvc files content")
        # 5. "find .dvc/cache -type f | head -10" returns (True, "cache content")
        # 6. "dvc status" returns (True, "dvc status output")
        # 7. "dvc status -r myremote" returns (True, "dvc remote status output")
        side_effects = [
            (True, "ignored"),               # Call 1
            (True, "config content"),        # Call 2
            (True, "config.local content"),  # Call 3
            (True, "dvc files content"),     # Call 4
            (True, "cache content"),         # Call 5
            (True, "dvc status output"),     # Call 6
            (True, "dvc remote status output"),  # Call 7
        ]
        mock_run_command.side_effect = side_effects

        temp_dir = "/tmp/fake"
        debug_dvc_setup(temp_dir)

        # Expected header and output log messages.
        expected_calls = [
            call("==== DVC DEBUGGING INFORMATION ===="),
            call("DVC config files:"),
            call("DVC config:"),
            call("config content"),
            call("DVC local config:"),
            call("config.local content"),
            call("DVC files created:"),
            call("dvc files content"),
            call("DVC cache directory:"),
            call("cache content"),
            call("DVC status:"),
            call("dvc status output"),
            call("DVC remote status:"),
            call("dvc remote status output"),
            call("===================================="),
        ]
        for expected in expected_calls:
            self.assertIn(expected, mock_logger.info.call_args_list)

    @patch("scripts.dvc_versioning.run_command")
    @patch("scripts.dvc_versioning.logger")
    def test_debug_dvc_setup_partial_failure(self, mock_logger, mock_run_command):
        # Set side effects:
        # 1. Check DVC config files returns (False, "").
        # 2. "cat .dvc/config" returns (False, "") -> no config output logged.
        # 3. "cat .dvc/config.local" returns (True, "") -> empty output.
        # 4. "find . -name '*.dvc'" returns (True, "dvc files content").
        # 5. "find .dvc/cache -type f | head -10" returns (False, "error in cache") -> not logged.
        # 6. "dvc status" returns (True, "") -> no status output.
        # 7. "dvc status -r myremote" returns (True, "remote status output").
        side_effects = [
            (False, ""),                     # Call 1
            (False, ""),                     # Call 2
            (True, ""),                      # Call 3
            (True, "dvc files content"),     # Call 4
            (False, "error in cache"),       # Call 5
            (True, ""),                      # Call 6
            (True, "remote status output"),  # Call 7
        ]
        mock_run_command.side_effect = side_effects

        temp_dir = "/tmp/fake"
        debug_dvc_setup(temp_dir)

        # Check that header messages are logged regardless of success.
        expected_header_calls = [
            call("==== DVC DEBUGGING INFORMATION ===="),
            call("DVC config files:"),
            call("DVC config:"),
            call("DVC local config:"),
            call("DVC files created:"),
            call("DVC cache directory:"),
            call("DVC status:"),
            call("DVC remote status:"),
            call("===================================="),
        ]
        for expected in expected_header_calls:
            self.assertIn(expected, mock_logger.info.call_args_list)

        self.assertIn(call("dvc files content"), mock_logger.info.call_args_list)
        self.assertIn(call("remote status output"), mock_logger.info.call_args_list)

        
        for logged_call in mock_logger.info.call_args_list:
            logged_message = logged_call[0][0]
            self.assertNotIn("config content", logged_message)
            self.assertNotIn("error in cache", logged_message)


    @patch("scripts.dvc_versioning.sys.exit")
    @patch("scripts.dvc_versioning.track_bucket_data")
    @patch("scripts.dvc_versioning.parse_arguments")
    @patch("scripts.dvc_versioning.logger")
    def test_main_success_debug_enabled(self, mock_logger, mock_parse_args, mock_track_bucket_data, mock_sys_exit):
        # Setup
        args = Namespace(
            cache_bucket="cache",
            dvc_remote="remote",
            debug=True,
            keep_temp=False,
            gcp_key_path="key",
            clear_remote=False,
        )
        mock_parse_args.return_value = args
        # Simulate a successful tracking operation.
        mock_track_bucket_data.return_value = True

        # Act
        main()

        # Assert
        mock_logger.setLevel.assert_called_once_with("DEBUG")
        mock_logger.info.assert_any_call("Verbose logging enabled")
        mock_track_bucket_data.assert_called_once_with("cache", "remote", True, False, "key", False)
        mock_sys_exit.assert_called_once_with(0)
        mock_logger.info.assert_any_call("DVC versioning completed successfully")


    @patch("scripts.dvc_versioning.sys.exit")
    @patch("scripts.dvc_versioning.track_bucket_data")
    @patch("scripts.dvc_versioning.parse_arguments")
    @patch("scripts.dvc_versioning.logger")
    def test_main_failure_debug_disabled(self, mock_logger, mock_parse_args, mock_track_bucket_data, mock_sys_exit):
        # Arrange: Create a Namespace with debug disabled and different options.
        args = Namespace(
            cache_bucket="cache",
            dvc_remote="remote",
            debug=False,
            keep_temp=True,
            gcp_key_path="key",
            clear_remote=True,
        )
        mock_parse_args.return_value = args
        # Simulate a failed tracking operation.
        mock_track_bucket_data.return_value = False

        # Act
        main()

        # Assert
        mock_logger.setLevel.assert_not_called()
        mock_track_bucket_data.assert_called_once_with("cache", "remote", False, True, "key", True)
        mock_sys_exit.assert_called_once_with(1)
        mock_logger.error.assert_any_call("DVC versioning failed")



if __name__ == "__main__":
    unittest.main()
