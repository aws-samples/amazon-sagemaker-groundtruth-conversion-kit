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

from skimage import io as skio
from skimage import img_as_ubyte
from skimage.color import rgba2rgb

from gt_conversion.convert_coco import CocoConverter


def test_segmentated_image():
    """
    Test a single image coco annotation
    """
    converter = CocoConverter()
    current_annotation_id = 5
    category_ids = {
        "(255, 127, 14)": 0,
        "(31, 119, 180)": 1,
        "(44, 160, 44)": 2,
    }

    img = img_as_ubyte(rgba2rgb(skio.imread("test/data/img1_annotated.png")))
    current_annotation_id, result = converter._annotate_single_image(
        image=img,
        image_id=0,
        category_ids=category_ids,
        current_annotation_id=current_annotation_id,
    )
    assert current_annotation_id == 7  # There are two annotations in this image
    print(result)


def test_segmentation_job_conversion(tmpdir):
    """
    This test will only pass with credentials for GT labeling job and S3 bucket.
    """
    job_name = "gt-converter-demo-job"
    converter = CocoConverter()
    converter.convert_job(job_name, output_coco_json_path=tmpdir + "output.json")

    with open(tmpdir + "output.json", "r") as outfile:
        print(outfile.readlines())


def test_boundingbox_job_conversion(tmpdir):
    """
    This test will only pass with credentials for GT labeling job and S3 bucket.
    """
    job_name = "gt-converter-demo-job-boundingbox"
    converter = CocoConverter()
    converter.convert_job(job_name, output_coco_json_path=tmpdir + "output.json")

    with open(tmpdir + "output.json", "r") as outfile:
        print(outfile.readlines())


def test_videotracking_job_conversion(tmpdir):
    """
    This test will only pass with credentials for GT labeling job and S3 bucket.
    """
    job_name = "MOT20example-clone"
    converter = CocoConverter()
    converter.convert_job(job_name, output_coco_json_path=tmpdir + "output.json")

    with open(tmpdir + "output.json", "r") as outfile:
        print(outfile.readlines())





test_videotracking_job_conversion("/tmp/")