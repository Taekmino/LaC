#!/usr/bin/env python3
"""VLM-based hazard detection and anxiety scoring node.

Uses GPT-4o as the Hazard Reasoner to identify potential navigation hazards,
and GPT-4o-mini as the Emotion Evaluator to assign anxiety scores (1-3) to
each detected hazard. Publishes hazardous object lists and anxiety scores
as JSON strings on ROS topics.
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import re
import time
import cv2
import json
import threading
import os
from cv_bridge import CvBridge, CvBridgeError

# Load .env file if available (for local development); in Docker, use environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import base64
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class MultiModal:
    """Minimal replacement for langchain_teddynote.models.MultiModal.

    Wraps a ChatOpenAI model to support image + text invocation
    using a system prompt and user prompt.
    """

    def __init__(self, llm, system_prompt="", user_prompt=""):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def invoke(self, image_path, display_image=False):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_data}},
            ]),
        ]
        response = self.llm.invoke(messages)
        return response.content

import tkinter as tk
from tkinter.scrolledtext import ScrolledText


class OutputGUI:
    """Tkinter GUI for displaying VLM inference results in real-time.

    Shows two text panels: one for hazard reasoner output and one for
    emotion evaluator output.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VLM Output Visualization")

        tk.Label(self.root, text="Hazard Reasoner Output", font=("Helvetica", 14, "bold")).pack(pady=(10,0))
        self.hazard_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20, font=("Helvetica", 12))
        self.hazard_text.pack(padx=10, pady=5)

        tk.Label(self.root, text="Emotion Evaluator Output", font=("Helvetica", 14, "bold")).pack(pady=(10,0))
        self.emotion_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=5, font=("Helvetica", 12))
        self.emotion_text.pack(padx=10, pady=5)

    def update_hazard(self, text):
        """Schedule a thread-safe update to the hazard text widget."""
        self.hazard_text.after(0, lambda: self._update_widget(self.hazard_text, text))

    def update_emotion(self, text):
        """Schedule a thread-safe update to the emotion text widget."""
        self.emotion_text.after(0, lambda: self._update_widget(self.emotion_text, text))

    def _update_widget(self, widget, text):
        """Replace the content of a ScrolledText widget."""
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()


class VLMProcessor:
    """ROS node that orchestrates VLM-based hazard detection and anxiety evaluation.

    Pipeline: image -> Hazard Reasoner (GPT-4o) -> hazardous objects + reasoning
    -> Emotion Evaluator (GPT-4o-mini) -> anxiety scores per object.
    Results are published on configurable ROS topics.
    """

    def __init__(self, gui=None):
        rospy.init_node('VLM_processor_node', anonymous=True)
        self.bridge = CvBridge()
        self.image = None
        self.hazard_reasoning = None
        self.hazard_processing = False
        self.prev_hazardous_objects = "None"
        self.gui = gui

        rospy.loginfo("Initializing VLM Processor...")

        # Configurable ROS parameters
        self.image_topic = rospy.get_param("~image_topic", "/viver1/front/left/color")
        self.hazard_topic_name = rospy.get_param("~hazardous_objects_topic", "/hazardous_objects")
        self.anxiety_topic_name = rospy.get_param("~anxiety_score_topic", "/anxiety_score")
        self.emotion_input_mode = rospy.get_param("~emotion_input_mode", "self_only")

        rospy.loginfo("Using image topic: %s", self.image_topic)
        rospy.loginfo("Using hazard topic: %s", self.hazard_topic_name)
        rospy.loginfo("Using anxiety topic: %s", self.anxiety_topic_name)
        rospy.loginfo("Emotion input mode: %s", self.emotion_input_mode)

        # Publishers
        self.hazard_pub = rospy.Publisher(self.hazard_topic_name, String, queue_size=10)
        self.anxiety_pub = rospy.Publisher(self.anxiety_topic_name, String, queue_size=10)

        # Subscriber
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)

        # Load prompt templates
        self.prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompt")
        rospy.loginfo("Loading prompt templates from: %s", self.prompt_dir)

        with open(os.path.join(self.prompt_dir, "hd_system_prompt_template.txt"), "r") as f:
            self.hd_system_prompt_template = f.read()
        with open(os.path.join(self.prompt_dir, "hd_user_prompt.txt"), "r") as f:
            self.hd_user_prompt = f.read()
        with open(os.path.join(self.prompt_dir, "ee_system_prompt_template.txt"), "r") as f:
            self.ee_system_prompt_template = f.read()
        with open(os.path.join(self.prompt_dir, "ee_user_prompt.txt"), "r") as f:
            self.ee_user_prompt = f.read()

        # Model names are configurable via ROS parameters (set in launch file)
        hd_model = rospy.get_param("~hazard_detection_model", "gpt-4o-2024-11-20")
        ee_model = rospy.get_param("~emotion_evaluator_model", "gpt-4o-mini")

        # Initialize Hazard Reasoner VLM
        rospy.loginfo("Setting up Hazard Detection VLM (model: %s)...", hd_model)
        self.llm_hd = ChatOpenAI(temperature=0.1, model_name=hd_model)
        self.hazard_detection = MultiModal(
            self.llm_hd,
            system_prompt=self.hd_system_prompt_template,
            user_prompt=self.hd_user_prompt
        )

        # Initialize Emotion Evaluator VLM
        rospy.loginfo("Setting up Emotion Evaluator VLM (model: %s)...", ee_model)
        self.llm_ee = ChatOpenAI(temperature=0.1, model_name=ee_model)
        self.emotion_evaluator = MultiModal(
            self.llm_ee,
            system_prompt=self.ee_system_prompt_template,
            user_prompt=self.ee_user_prompt
        )

        rospy.loginfo("VLM Processor initialization complete!")

    def image_callback(self, msg):
        """Receives RGB images and triggers hazard detection in a background thread."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image = cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))
            return

        if not self.hazard_processing:
            self.hazard_processing = True
            threading.Thread(target=self.process_hazard).start()

    def clean_json_output(self, result):
        """Strips markdown code fences from VLM JSON output."""
        cleaned_result = re.sub(r"```json\s*", "", result)
        cleaned_result = re.sub(r"```", "", cleaned_result)
        return cleaned_result.strip()

    def process_hazard(self):
        """Runs the Hazard Reasoner VLM on the latest image.

        Saves the image to a temp file, invokes the VLM with the hazard reasoning
        prompt, parses the JSON output, and publishes the hazardous objects list.
        If hazards are found, triggers the emotion evaluator.
        """
        if self.image is None:
            self.hazard_processing = False
            return
        hazard_input_image = self.image.copy()
        temp_image_path = "/tmp/current_image.jpg"
        cv2.imwrite(temp_image_path, hazard_input_image)

        hd_updated_system_prompt = self.hd_system_prompt_template.replace(
            "{prev_hazardous_objects}",
            str(self.prev_hazardous_objects)
        )
        self.hazard_detection.system_prompt = hd_updated_system_prompt

        start_time = time.time()
        result = self.hazard_detection.invoke(temp_image_path, display_image=False)
        rospy.loginfo("Hazard detection result: {}".format(result))
        end_time = time.time()
        rospy.loginfo("Hazard detection inference time: {:.2f} seconds".format(end_time - start_time))

        cleaned_result = self.clean_json_output(result)

        try:
            result_json = json.loads(cleaned_result)
            self.hazard_reasoning = result_json.get("hazard_reasoning", None)
            hazardous_objects = result_json.get("hazardous_objects", [])
            self.hazard_pub.publish(json.dumps(hazardous_objects))
            rospy.loginfo("Risk analysis saved and hazardous_objects published")

            if hazardous_objects and len(hazardous_objects) > 0:
                self.prev_hazardous_objects = hazardous_objects
            else:
                self.prev_hazardous_objects = "None"

            if self.gui:
                description = result_json.get("description", "None")
                object_list = result_json.get("object_list", "None")
                hazard_text = (
                    "Description:\n" + str(description) + "\n\n" +
                    "Object List:\n" + str(object_list) + "\n\n" +
                    "Hazard Reasoning:\n" + str(self.hazard_reasoning) + "\n\n" +
                    "Hazardous Objects:\n" + str(hazardous_objects)
                )
                self.gui.update_hazard(hazard_text)

        except Exception as e:
            rospy.logerr("JSON parsing error in hazard detection: {}".format(e))
            self.hazard_processing = False
            return

        if self.hazard_reasoning is not None and hazardous_objects:
            threading.Thread(target=self.process_emotion, args=(hazard_input_image, self.hazard_reasoning, hazardous_objects)).start()

        self.hazard_processing = False

    def process_emotion(self, hazard_input_image, hazard_reasoning, hazardous_objects):
        """Runs the Emotion Evaluator VLM to assign anxiety scores to hazards.

        Takes the hazard reasoning output and detected objects, optionally combines
        the original and current images, and invokes the emotion evaluator VLM.
        Publishes anxiety scores as a JSON dictionary.
        """
        if isinstance(hazard_reasoning, list):
            hazard_reasoning_str = ", ".join(hazard_reasoning)
        else:
            hazard_reasoning_str = hazard_reasoning if hazard_reasoning is not None else "None"

        ee_updated_system_prompt = self.ee_system_prompt_template.replace(
            "{hazard_reasoning}",
            hazard_reasoning_str
        )
        ee_updated_system_prompt = ee_updated_system_prompt.replace(
            "{hazardous_objects}",
            json.dumps(hazardous_objects) if hazardous_objects else "None"
        )
        self.emotion_evaluator.system_prompt = ee_updated_system_prompt

        if self.emotion_input_mode == "self_and_current" and self.image is not None:
            current_image = self.image.copy()
            combined_image = cv2.hconcat([hazard_input_image, current_image])
            emotion_input_image = combined_image
        else:
            emotion_input_image = hazard_input_image

        temp_image_path = "/tmp/emotion_image.jpg"
        cv2.imwrite(temp_image_path, emotion_input_image)

        start_time = time.time()
        result_emotion = None
        while result_emotion is None:
            try:
                result_emotion = self.emotion_evaluator.invoke(temp_image_path, display_image=False)
            except Exception as e:
                if "Rate limit" in str(e):
                    time.sleep(5)
                else:
                    time.sleep(1)
        end_time = time.time()
        rospy.loginfo("Emotion evaluator result: {}".format(result_emotion))
        rospy.loginfo("Emotion evaluator inference time: {:.2f} seconds".format(end_time - start_time))

        cleaned_emotion_result = self.clean_json_output(result_emotion)

        try:
            emotion_result_json = json.loads(cleaned_emotion_result)
            anxiety_score = emotion_result_json.get("anxiety_score", {})
            self.anxiety_pub.publish(json.dumps(anxiety_score))
            rospy.loginfo("Anxiety score published: {}".format(json.dumps(anxiety_score)))
            if self.gui:
                emotion_text = "Anxiety Score:\n" + str(anxiety_score)
                self.gui.update_emotion(emotion_text)
        except Exception as e:
            pass

    def spin(self):
        """Blocks until ROS shutdown."""
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.loginfo("Starting VLM Processor application...")
        gui = OutputGUI()
        node = VLMProcessor(gui=gui)
        rospy.loginfo("Starting ROS thread...")
        ros_thread = threading.Thread(target=node.spin)
        ros_thread.daemon = True
        ros_thread.start()
        rospy.loginfo("Starting GUI main loop...")
        gui.run()
    except Exception as e:
        rospy.logerr("Error in main function: %s", str(e))
    finally:
        rospy.loginfo("VLM Processor application shutting down...")
