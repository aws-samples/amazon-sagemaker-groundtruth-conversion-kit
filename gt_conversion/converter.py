import os
import abc
import json
import boto3

from gt_conversion.utils import split_s3_bucket_key


class Converter(abc.ABC):
    """
    Abstract base class for data format converters.
    """

    def __init__(self):
        self.sm_client = boto3.client("sagemaker")
        self.s3_client = boto3.client("s3")

    @abc.abstractmethod
    def convert_job(self, job_name, output_coco_json_path):
        pass

    def _maybe_download_from_s3(self, manifest_path):
        """
        Downloads manifest local disk if s3 path is specified. If the manifest was downloaded from S3
        then a flag will be returned so the local file can be removed after conversion.
        :param manifest_path: Path of manifest file
        :return: manifest_path(str), cleanup_flag(bool)
        """
        if "s3://" == manifest_path[:5]:
            bucket, key = split_s3_bucket_key(manifest_path)

            with open("local_manifest", "wb") as f:
                self.s3_client.download_fileobj(bucket, key, f)
                manifest_path = "local_manifest"

            # counting lines for progressbar
            with open("local_manifest") as f:
                self.manifestcount = len(list(f))

            return manifest_path, True

        else:
            return manifest_path, False

    def manifest_reader(self, manifest_path):
        """
        Generator to return each image GT annotations
        :param manifest_path: Path of manifest file
        """
        manifest_path, cleanup = self._maybe_download_from_s3(manifest_path)

        for line in open(manifest_path, mode="r"):
            yield json.loads(line)

        if cleanup:
            os.remove(manifest_path)
