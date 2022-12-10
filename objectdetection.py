import logging
import os
import queue
import random
import time
import cv2
import numpy as np
import carla
import tensorflow as tf

# We want to run a custom tensorflow object detection model in Carla
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
        model_path: Path to the model directory.

    Returns:
        Object detection model.
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

def generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_vehicles):
    """Generate traffic"""
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_random_device_seed(0)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.global_percentage_speed_difference(30.0)
    vehicle_bp = blueprint_library.filter('vehicle.*')
    spawn_points = spawn_points
    number_of_spawn_points = len(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []
    vehicles_list = []

    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles:
            break
        blueprint = random.choice(vehicle_bp)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        # spawn
        #print("spawned")

        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, False):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    return vehicles_list



def main(model_path, town, num_vehicles, num_frames, ):
    # Setup world and spawn ego
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    #world = client.get_world()
    world = client.load_world(town)
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
    image = image_queue.get()

    # Generate traffic
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    vehicles_list = generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_vehicles)

    # Load model and labels and create detection function
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

    # Main Game Loop, run until we reach the desired number of frames
    while image.frame < num_frames:
        world.tick()
        # Get image
        image = image_queue.get()
        # Get image as numpy array with shape (600, 800, 3)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        reshaped_array = np.reshape(array, (image.height, image.width, 4))
        array = reshaped_array[:, :, :3]
        array = array[:, :, ::-1]

        # Perform detection
        t0 = time.perf_counter()
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
            min_score_thresh=.3,
            agnostic_mode=False)
        t1 = time.perf_counter()
        print(f"Time to run inference: {t1 - t0:0.4f} seconds")

        # Display image
        # convert cv2 image to rbg image
        cv2_im = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
        cv2.imshow('RaceAI RT-Classification',cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':


    model_path = 'models/saved_model_v2/'
    num_vehicles = 75
    num_frames = 10000

    main(model_path, "Town10HD", num_vehicles, num_frames)
