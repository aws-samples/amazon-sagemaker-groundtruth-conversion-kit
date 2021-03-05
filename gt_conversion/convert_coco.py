import json
import io
import ast

import numpy as np
from skimage import measure
from skimage import io as skio
from skimage import img_as_ubyte
from skimage.color import rgba2rgb
from shapely.geometry import Polygon, MultiPolygon

from gt_conversion.converter import Converter
from gt_conversion.utils import split_s3_bucket_key
import tqdm


class CocoConverter(Converter):
    """
    Converter from GT format to COCO standard.
    """

    def __init__(self):
        super().__init__()
        self.background_color = (255, 255, 255)

    def _build_category_ids(self, manifest_path, job_name, background_color):
        """
        Build a dictionary mapping label ids to annotation RBG colors
        :param manifest_path: Path to manifest ffile
        :param job_name: SageMaker gt jobname
        :param background_color: RGB tuple for background color
        :return: Dictionary of label ids mapped to RBG colors
        """
        metadata_key = job_name + "-ref-metadata"
        colors_found = set()
        for label in self.manifest_reader(manifest_path):
            for i in label[metadata_key]["internal-color-map"].keys():
                colors_found.add(
                    label[metadata_key]["internal-color-map"][i]["hex-color"]
                )

        # Convert to RGB
        colors_found = [
            tuple(int(s.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            for s in list(colors_found)
        ]
        colors_found.remove(background_color)

        # Construct color:label dict
        num_labels = len(colors_found)  # one of these is the background color

        # Define which colors match which categories in the images
        category_ids = {}
        for i in range(num_labels):
            category_ids[str(colors_found[i])] = i  # TODO: Write as COCO format

        return category_ids

    @staticmethod
    def _create_submasks(annotated_image, category_ids):
        """
        Create masks for each labels annotation label
        :param annotated_image: Numpy array of annotated image
        :param category_ids: Dictionary of label ids mapped to RGB values
        :return: Dictionary of numpy arrays for each labels annotations
        """
        sub_masks = {}

        for color in category_ids.keys():
            color = ast.literal_eval(color)
            sub_mask = np.pad(
                np.alltrue(annotated_image == list(color), axis=2),
                pad_width=1,
                mode="constant",
                constant_values=False,
            )

            if np.sum(sub_mask) > 0:
                sub_masks[str(color)] = sub_mask

        return sub_masks

    @staticmethod
    def _create_submask_annotation(sub_mask, image_id, category_id, annotation_id):
        """
        Find contours (boundary lines) around each sub-mask
        Note: there could be multiple contours if the object is partially occluded. (E.g. an elephant behind a tree)
        :param sub_mask: Numpy array of single labels masks
        :param image_id: Integer ID specifying which image this is
        :param category_id: Label ID being processed
        :param annotation_id: Integer ID specifying which annotation this is
        :return: Dictionary of COCO annotation
        """
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation="low")

        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y) and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

        annotation = {
            "segmentation": segmentations,
            "iscrowd": 0,
            "image_id": image_id,
            "category_id": category_id,
            "id": annotation_id,
            "bbox": bbox,
            "area": area,
        }

        return annotation

    def _annotate_single_image(
        self, image, image_id, category_ids, current_annotation_id
    ):
        """
        Create the annotations for a single image file
        :param image: Numpy array of annotated image
        :param image_id: Integer ID specifying which image this is
        :param category_ids: Dictionary of label ids mapped to annotation RBG colors
        :param current_annotation_id: The current annotation id of the manifest being converted
        :return: List of image annotations
        """
        annotations = []
        sub_masks = self._create_submasks(image, category_ids)

        for color, sub_mask in sub_masks.items():
            category_id = category_ids[color]
            submask_annotation = self._create_submask_annotation(
                sub_mask, image_id, category_id, current_annotation_id
            )
            annotations.append(submask_annotation)
            current_annotation_id += 1

        return current_annotation_id, annotations

    def _convert_segmentation_manifest(
        self, manifest_path, job_name, output_coco_json_path
    ):
        """
        Converts a single segmentation manifest file into COCO format.
        :param manifest_path: Path of the GT manifest file
        :param job_name: Name of the GT job
        :param output_coco_json_path: Output path for converted COCO json.
        """
        category_ids = self._build_category_ids(
            manifest_path, job_name, self.background_color
        )
        image_id = 0
        current_annotation_id = 0
        annotations = []
        images = []

        for annotation in tqdm.tqdm(
            self.manifest_reader(manifest_path), total=self.manifestcount
        ):
            bucket, key = split_s3_bucket_key(annotation[job_name + "-ref"])

            outfile = io.BytesIO()
            self.s3_client.download_fileobj(bucket, key, outfile)
            outfile.seek(0)

            img_annotated = img_as_ubyte(rgba2rgb(skio.imread(outfile)))
            current_annotation_id, img_annotations = self._annotate_single_image(
                img_annotated, image_id, category_ids, current_annotation_id
            )
            w, h, c = img_annotated.shape
            images.append(
                {
                    "file_name": annotation["source-ref"],
                    "height": h,
                    "width": w,
                    "id": image_id,
                }
            )
            annotations.extend(img_annotations)
            image_id += 1

        coco_json = {
            "type": "instances",
            "images": images,
            "categories": category_ids,
            "annotations": annotations,
        }

        with open(output_coco_json_path, "w") as f:
            json.dump(coco_json, f)

    def _convert_bbox_manifest(self, manifest_path, job_name, output_coco_json_path):
        """
        Converts a single bounding box manifest file into COCO format.
        :param manifest_path: Path of the GT manifest file
        :param job_name: Name of the GT job
        :param output_coco_json_path: Output path for converted COCO json.
        """
        image_id = 0
        annotation_id = 0
        annotations = []
        images = []
        category_ids = {}

        for annotation in self.manifest_reader(manifest_path):
            w = annotation[job_name]["image_size"][0]["width"]
            h = annotation[job_name]["image_size"][0]["height"]
            images.append(
                {
                    "file_name": annotation["source-ref"],
                    "height": h,
                    "width": w,
                    "id": image_id,
                }
            )

            for bbox in annotation[job_name]["annotations"]:
                coco_bbox = {
                    "iscrowd": 0,
                    "image_id": image_id,
                    "category_id": bbox["class_id"],
                    "id": annotation_id,
                    "bbox": (bbox["left"], bbox["top"], bbox["width"], bbox["height"]),
                    "area": bbox["width"] * bbox["height"],
                }
                annotations.append(coco_bbox)
                annotation_id += 1
                category_ids.update(
                    annotation[job_name + "-metadata"]["class-map"]
                )  # TODO: Write as COCO format

            image_id += 1

        coco_json = {
            "type": "instances",
            "images": images,
            "categories": category_ids,
            "annotations": annotations,
        }

        with open(output_coco_json_path, "w") as f:
            json.dump(coco_json, f)

    def _convert_video_tracking_manifest(self, manifest_path, job_name, output_coco_json_path):
        """
        Converts a single video tracking manifest file into COCO format.
        :param manifest_path: Path of the GT manifest file
        :param job_name: Name of the GT job
        :param output_coco_json_path: Output path for converted COCO json.
        """
        # image_id = 0 # -> frame_id 
        seq_id = 1 # sequence_id, starts from 1 in the GT input manifest
        sequences = {}

        # assuming each output manifest for GT points to one SeqLabel.json which is what we need
        
        for output_manifest in self.manifest_reader(manifest_path):
            seq_label_path = output_manifest[job_name+'-ref']
            print("\nProcessing sequence: "+str(seq_id))
            print("Frames: ", end='')
            category_ids = {}
            sequences["sequence-"+str(seq_id)]=[]

            for frame in self.tracking_manifest_reader(seq_label_path):
                
                annotation_id = 0
                annotations = []
                images = []
                frame_id = frame["frame-no"]
                file_name= frame["frame"]
                print(frame_id, end=' ')
                

                for annotation in frame["annotations"]:

                    w = annotation["width"]
                    h = annotation["height"]
                    #
                    images.append(
                        {
                            "file_name": file_name,
                            "height": h,
                            "width": w,
                            "id": frame_id,
                        }
                    )

                    #
                    coco_bbox = {
                        "iscrowd": 0,
                        "image_id": frame_id,
                        "category_id": annotation["class-id"],
                        "id": annotation["object-id"],
                        "bbox": (annotation["left"], annotation["top"], annotation["width"], annotation["height"]),
                        "area": annotation["width"] * annotation["height"],
                    }

                    annotations.append(coco_bbox)
                    annotation_id += 1
                    category_ids.update(
                        {"supercategory": annotation["object-name"].split(":")[0],
                        "id":annotation["object-id"],
                        "name":annotation["object-name"]}
                    )  # TODO: Verify if most common format is COCO for object tracking?



                coco_json = {
                    "type": "instances",
                    "images": images,
                    "categories": category_ids,
                    "annotations": annotations,
                } 

                sequences["sequence-"+str(seq_id)].append(coco_json) 
            seq_id+=1 

        with open(output_coco_json_path, "w") as f:
            json.dump(sequences, f)


    def convert_job(self, job_name, output_coco_json_path):
        """
        Converts a SageMaker Ground Truth job's manifest file to COCO format.
        :param job_name: Name of the GT job (str)
        :param output_coco_json_path: Path to write output file
        """
        job_description = self.sm_client.describe_labeling_job(LabelingJobName=job_name)
        job_state = job_description["LabelingJobStatus"]
        job_task_keywords = job_description["HumanTaskConfig"]["TaskKeywords"]
        

        if job_state == "Completed":
            if "Images" not in job_task_keywords and "Video" not in job_task_keywords:
                raise NotImplementedError(
                    "GT Conversion only supports image and video labeling tasks at the moment."
                )

            manifest_path = job_description["LabelingJobOutput"]["OutputDatasetS3Uri"]

            if "bounding boxes" in job_task_keywords:
                self._convert_bbox_manifest(
                    manifest_path, job_name, output_coco_json_path
                )
            elif "image segmentation" in job_task_keywords:
                self._convert_segmentation_manifest(
                    manifest_path, job_name, output_coco_json_path
                )
            
            elif "Video" in job_task_keywords and "tracking" in job_task_keywords:
                self._convert_video_tracking_manifest(
                    manifest_path, job_name, output_coco_json_path
                )

            else:
                raise ValueError(
                    "Could not determine the type of labeling job. Currently, Bounding Box and Semantic Segmentation are supported."
                )

        else:
            raise ValueError(
                "Job is not in `Completed` state. Currently: {}".format(job_state)
            )