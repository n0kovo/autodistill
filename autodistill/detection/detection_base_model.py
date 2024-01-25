import datetime
import glob
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import roboflow
import supervision as sv
from PIL import ImageFile
from supervision.utils.file import save_text_file
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.helpers import load_image, split_data

from .detection_ontology import DetectionOntology


@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Detections:
        pass

    def sahi_predict(self, input: str) -> sv.Detections:
        slicer = sv.InferenceSlicer(callback=self.predict)

        return slicer(load_image(input, return_format="cv2"))

    def _record_confidence_in_files(
        self,
        annotations_directory_path: str,
        images: Dict[str, np.ndarray],
        annotations: Dict[str, sv.Detections],
    ) -> None:
        Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
        for image_name, _ in images.items():
            detections = annotations[image_name]
            yolo_annotations_name, _ = os.path.splitext(image_name)
            confidence_path = os.path.join(
                annotations_directory_path,
                "confidence-" + yolo_annotations_name + ".txt",
            )
            confidence_list = [str(x) for x in detections.confidence.tolist()]
            save_text_file(lines=confidence_list, file_path=confidence_path)
            print("Saved confidence file: " + confidence_path)
            
    def _process_directory(
        self,
        directory,
        output_folder,
        load_truncated_images,
        sahi,
        with_nms,
        record_confidence,
        human_in_the_loop,
        roboflow_project
        ):
        
        images_map = {}
        detections_map = {}

        files = glob.glob(os.path.join(directory, '*'))
        progress_bar = tqdm(files, desc=f"Processing directory: {directory}")

        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

            try:
                image = cv2.imread(f_path)
                if image is None and load_truncated_images:
                    continue
            except Exception as e:
                raise

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()

            try:
                if sahi:
                    detections = slicer(f_path)
                else:
                    detections = self.predict(f_path)

                if with_nms:
                    detections = detections.with_nms()
            except OSError as e:
                progress_bar.write(f"Could not process image '{f_path}': {e}")
                continue

        # Write metadata after processing each directory
        if detections_map:
            dataset = sv.DetectionDataset(self.ontology.classes(), images_map, detections_map)
            dataset.as_yolo(output_folder + "/images", output_folder + "/annotations", 
                            min_image_area_percentage=0.01, data_yaml_path=output_folder + "/data.yaml")
            if record_confidence:
                self._record_confidence_in_files(output_folder + "/annotations", images_map, detections_map)
            
            if human_in_the_loop:
                roboflow.login()
                rf = roboflow.Roboflow()
                workspace = rf.workspace()
                workspace.upload_dataset(output_folder, project_name=roboflow_project)
                
            split_data(output_folder, record_confidence=record_confidence)
        else:
            raise RuntimeError(f"No images could be processed in directory {directory}")

        return images_map, detections_map
    
    
    def _process_images(
        self,
        input_folder,
        extensions,
        recursive,
        output_folder,
        load_truncated_images,
        sahi,
        with_nms,
        record_confidence,
        human_in_the_loop,
        roboflow_project
        ):
        
        os.makedirs(output_folder, exist_ok=True)

        # Initialize combined maps
        combined_images_map = {}
        combined_detections_map = {}

        # Find directories
        directories = {os.path.dirname(file) for ext in extensions for file in glob.glob(f"{input_folder}/**/*{ext}", recursive=recursive)}

        for directory in directories:
            images_map, detections_map = self._process_directory(directory, output_folder, load_truncated_images, sahi, with_nms, record_confidence, human_in_the_loop, roboflow_project)
            combined_images_map.update(images_map)
            combined_detections_map.update(detections_map)

        # Create the combined dataset
        dataset = sv.DetectionDataset(self.ontology.classes(), combined_images_map, combined_detections_map)
        
        if not combined_detections_map:
            raise RuntimeError("No images could be processed")

        # Perform the final dataset operations
        dataset.as_yolo(output_folder + "/images", output_folder + "/annotations", 
                        min_image_area_percentage=0.01, data_yaml_path=output_folder + "/data.yaml")
        if record_confidence:
            self._record_confidence_in_files(output_folder + "/annotations", combined_images_map, combined_detections_map)
        split_data(output_folder, record_confidence=record_confidence)

        return dataset


    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        extensions: list = None,
        recursive: bool = False,
        load_truncated_images: bool = False,
        output_folder: str = None,
        human_in_the_loop: bool = False,
        roboflow_project: str = None,
        roboflow_tags: str = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        with_nms: bool = False,
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        
      # Use 'extensions' if set. Fall back to 'extension'
        if extensions is not None:
            if extension != ".jpg":
                raise ValueError("`extension` and `extensions` are mutually exclusive.")
        else:
            extensions = [extension]

        dataset = self._process_images(
            input_folder=input_folder,
            extensions=extensions,
            recursive=recursive,
            output_folder=output_folder,
            load_truncated_images=load_truncated_images,
            sahi=sahi,
            with_nms=with_nms,
            record_confidence=record_confidence,
            human_in_the_loop=human_in_the_loop,
            roboflow_project=roboflow_project
            )
        
        print("Labeled dataset created - ready for distillation.")
        return dataset
