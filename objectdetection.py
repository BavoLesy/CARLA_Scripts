import os
import queue
import random
import sys
import time

import cv2
import numpy as np
import platform

from PIL import Image
from PIL import ImageDraw
import carla
import pygame
import tensorflow as tf
# We want to run a custom tensorflow object detection model in Carla using
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn

def load_model(model_path):
    """Loads the model.

    Args:
        model_path: Path to the .tflite file.

    Returns:
        A tuple of (interpreter, input_details, output_details).
    """
    pipeline_config = model_path + 'pipeline.config'
    model_dir = model_path + 'checkpoint/ckpt-0'
    # Load pipeline config and build a detection model.
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir)).expect_partial()

    return detection_model


def run_inference(interpreter, image, input_details, output_details):
    """Runs inference on an input image.

    Args:
        interpreter: Interpreter object.
        image: Input image.

    Returns:
        A tuple of (boxes, classes, scores).
    """
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    # Run inference.
    #size = np.ones(1, dtype=np.dtype("uint8"))
    #set size to uint8
    #interpreter.set_tensor(input_details[0]['index'], size)

    #interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    # Post-processing: remove batch dimension and find the detections with
    # confidence score above the threshold.
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    #count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    return boxes, classes, scores

def main(model_path):
    # Setup world and spawn ego
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    ego = world.spawn_actor(bp, random.choice(spawn_points))
    ego.set_autopilot(True)
    # Sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05  # (20fps)
    world.apply_settings(settings)
    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego)
    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    world.tick()
    # Get image
    image = image_queue.get()
    image_width, image_height = 600, 400
    # Start pygame
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    detection_model = load_model(model_path)
    detect_fn = get_model_detection_function(detection_model)
    label_map_pbtxt_fname = model_path + 'labelmap.pbtxt'
    label_map_path = label_map_pbtxt_fname
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)


    while True:
        clock.tick(20)
        world.tick()
        # Get image
        image = image_queue.get()
        # Get image as numpy array with shape (600, 800, 3)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        #Create a PIL image from the array
        pil_image = Image.fromarray(array.reshape((image.height, image.width, 4)))
        #Convert the PIL image to RGB
        pil_image = pil_image.convert('RGB')
        reshaped_array = np.reshape(array, (image.height, image.width, 4))
        array = reshaped_array[:, :, :3]
        array = array[:, :, ::-1]


        input_tensor = tf.convert_to_tensor(np.expand_dims(array, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = array.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=20,
            min_score_thresh=.1,
            agnostic_mode=False)
        # show image as rgb
        cv2.imshow('image', image_np_with_detections)
        #cv2.imshow('name' , image_np_with_detections)
        #boxes, classes, scores = run_inference(interpreter, array, input_details, output_details)
        #print(boxes, classes, scores)
        #for i in range(len(boxes)):
        #    if scores[i] > 0.5:
        #        box = boxes[i]
        #        x1 = int(box[1] * image_width)
        #        y1 = int(box[0] * image_height)
        #        x2 = int(box[3] * image_width)
        #        y2 = int(box[2] * image_height)
        #        pygame.draw.rect(display, colors[classes[i]], (x1, y1, x2 - x1, y2 - y1), 2)
        #        display.blit(font.render(labels[classes[i]], True, colors[classes[i]]), (x1, y1))
        #pygame.display.flip()
        # Convert the PIL image to a numpy array
        image_np = np.array(pil_image)
        # Display the resulting frame
        #cv2.imshow('frame', image_np_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Run inference
        t0 = time.perf_counter()
        # Get image from array


        np_image = np.asarray(pil_image)



        pygame.display.set_caption('Carla Object Detection')
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        # Run inference





if __name__ == '__main__':
    main('models/saved_model_v1/')

