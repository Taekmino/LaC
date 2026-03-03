#!/usr/bin/env python3
"""Zero-shot segmentation node using GroundingDINO + EdgeSAM.

Subscribes to hazardous object lists and anxiety scores from the VLM pipeline,
detects and segments hazardous objects in real-time RGB images, and publishes
a segmentation mask where pixel values encode anxiety levels (253-255 for
anxiety scores 1-3, 251 for detected but unscored objects).
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
import ast
import json
import torch
import torchvision
import sys

# Add EdgeSAM to the Python path
EDGE_SAM_DIR = os.getenv("EDGE_SAM_DIR", "/home/appuser/LaC/EfficientSAM")
sys.path.append(EDGE_SAM_DIR)

from groundingdino.util.inference import Model
from edge_sam import SamPredictor, build_edge_sam

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths (configurable via environment variables)
GROUNDING_DINO_CONFIG_PATH = os.getenv(
    "GROUNDING_DINO_CONFIG_PATH",
    "/home/appuser/LaC/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.getenv(
    "GROUNDING_DINO_CHECKPOINT_PATH",
    "/home/appuser/LaC/groundingdino_swint_ogc.pth"
)
EDGE_SAM_CHECKPOINT_PATH = os.getenv(
    "EDGE_SAM_CHECKPOINT_PATH",
    "/home/appuser/LaC/EfficientSAM/edge_sam_3x.pth"
)

# Detection and segmentation hyperparameters
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.8
NMS_THRESHOLD = 0.8


class GroundedSegmentationNode:
    """ROS node for zero-shot object detection and segmentation.

    Uses GroundingDINO for text-guided object detection and EdgeSAM for
    mask generation. Combines detected masks into a single mono8 image
    where pixel values encode the anxiety score for each hazardous region.
    """

    def __init__(self):
        rospy.init_node('grounded_segmentation_node', anonymous=True)
        self.bridge = CvBridge()
        self.image = None
        self.image_stamp = None

        self.hazardous_list = []
        self.anxiety_dict = {}

        # Subscribers
        rospy.Subscriber("/viver1/front/left/color", Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/hazardous_objects", String, self.hazardous_callback, queue_size=1)
        rospy.Subscriber("/anxiety_score_topic", String, self.anxiety_callback, queue_size=1)

        # Publisher for combined segmentation mask
        self.mask_pub = rospy.Publisher("/segmentation_mask", Image, queue_size=1)

        # Initialize GroundingDINO model
        rospy.loginfo("Initializing GroundingDINO model...")
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
        )

        # Initialize EdgeSAM model
        rospy.loginfo("Initializing EdgeSAM model...")
        self.edge_sam = build_edge_sam(checkpoint=EDGE_SAM_CHECKPOINT_PATH)
        self.edge_sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(self.edge_sam)

    def hazardous_callback(self, msg):
        """Updates the list of hazardous objects to detect from the VLM pipeline."""
        try:
            self.hazardous_list = ast.literal_eval(msg.data)
            rospy.loginfo("Updated hazardous list: %s", self.hazardous_list)
        except Exception as e:
            rospy.logerr("Failed to parse hazardous objects: %s", e)

    def anxiety_callback(self, msg):
        """Updates the anxiety score dictionary from the emotion evaluator."""
        try:
            new_scores = json.loads(msg.data)
            for key, value in new_scores.items():
                self.anxiety_dict[key] = value
            rospy.loginfo("Updated anxiety dictionary: %s", self.anxiety_dict)
        except Exception as e:
            rospy.logerr("Failed to parse anxiety score: %s", e)

    def image_callback(self, msg):
        """Receives RGB images and triggers detection+segmentation if hazards are known."""
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_stamp = msg.header.stamp
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)
            return

        if not self.hazardous_list:
            rospy.logwarn("Hazardous list is empty. Skipping detection.")
            return

        self.process_detection_segmentation()

    def process_detection_segmentation(self):
        """Detects and segments hazardous objects, then publishes the anxiety-encoded mask.

        Detection: GroundingDINO predicts bounding boxes from text prompts.
        Segmentation: EdgeSAM generates masks for each detected box.
        Each mask pixel is assigned a value encoding its anxiety level:
          - 251: detected but no anxiety score assigned yet
          - 253: anxiety score 1
          - 254: anxiety score 2
          - 255: anxiety score 3
        Overlapping masks use max-fusion (highest pixel value wins).
        """
        if self.image is None:
            return

        start_time = time.time()

        # Detect objects using GroundingDINO with hazardous object names as prompts
        detections = self.grounding_dino_model.predict_with_classes(
            image=self.image,
            classes=self.hazardous_list,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        if detections.xyxy.size > 0:
            rospy.loginfo("Number of boxes before NMS: %d", len(detections.xyxy))
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            rospy.loginfo("Number of boxes after NMS: %d", len(detections.xyxy))
        else:
            rospy.loginfo("No objects detected.")
            return

        # Segment each detected object using EdgeSAM
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        height, width = self.image.shape[:2]
        final_mask = np.zeros((height, width), dtype=np.uint8)

        for idx, box in enumerate(detections.xyxy):
            _masks, scores, logits = self.sam_predictor.predict(
                box=box,
                num_multimask_outputs=1
            )
            selected_mask = _masks[np.argmax(scores)]
            binary_mask = selected_mask.astype(np.uint8)

            # Map detection to object label
            try:
                label_idx = int(detections.class_id[idx])
                if label_idx < len(self.hazardous_list):
                    object_label = self.hazardous_list[label_idx]
                else:
                    object_label = "unknown"
            except Exception as e:
                rospy.logwarn("Failed to get object label for detection %d: %s", idx, e)
                object_label = "unknown"

            # Encode anxiety score as pixel value
            if object_label in self.anxiety_dict:
                try:
                    anxiety_score = int(self.anxiety_dict[object_label])
                except (ValueError, TypeError):
                    anxiety_score = 0
                pixel_value = 252 + anxiety_score
            else:
                pixel_value = 251

            object_mask = binary_mask * pixel_value
            final_mask = np.maximum(final_mask, object_mask)

        end_time = time.time()
        rospy.loginfo("Detection and segmentation inference time: %.3f sec", (end_time - start_time))

        # Publish combined mask with original image timestamp
        mask_msg = self.bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
        mask_msg.header.stamp = self.image_stamp
        self.mask_pub.publish(mask_msg)

    def spin(self):
        """Blocks until ROS shutdown."""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = GroundedSegmentationNode()
        rospy.loginfo("Grounded Segmentation Node Started.")
        node.spin()
    except rospy.ROSInterruptException:
        pass
