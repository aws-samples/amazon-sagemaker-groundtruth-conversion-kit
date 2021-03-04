# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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

    def _maybe_download_from_s3(self, manifest_path, localpathname="local_manifest"):
        """
        Downloads manifest local disk if s3 path is specified. If the manifest was downloaded from S3
        then a flag will be returned so the local file can be removed after conversion.
        :param manifest_path: Path of manifest file
        :return: manifest_path(str), cleanup_flag(bool)
        """
        if "s3://" == manifest_path[:5]:
            bucket, key = split_s3_bucket_key(manifest_path)

            with open(localpathname, "wb") as f:
                self.s3_client.download_fileobj(bucket, key, f)
                manifest_path = localpathname
            
            # counting lines for progressbar
            with open(localpathname) as f:
                count = len(list(f))
            
            return manifest_path, True, count

        else:
            return manifest_path, False

    def manifest_reader(self, manifest_path, localpathname="output_manifest"):
        """
        Generator to return each image GT annotations
        :param manifest_path: Path of manifest file
        """
        manifest_path, cleanup, self.manifestcount = self._maybe_download_from_s3(manifest_path,localpathname)

        for line in open(manifest_path, mode="r"):
            yield json.loads(line)

        if cleanup:
            os.remove(manifest_path)

    def tracking_manifest_reader(self, manifest_path, localpathname="tracking_manifest"):
        """
        Generator to return annotationsf from each sequence of frames
        :param manifest_path: Path of manifest file
        """

        manifest_path, cleanup, self.manifestcount = self._maybe_download_from_s3(manifest_path,localpathname)

        for annotation in json.load(open(manifest_path, mode="r"))['tracking-annotations']:
            yield annotation

        if cleanup:
            os.remove(manifest_path)


