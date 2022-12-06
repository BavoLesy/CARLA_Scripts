import logging

import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


# Calculate 2D projection of 3D coordinate
def get_image_point(loc, K, w2c):
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]


def generate_traffic(traffic_manager, client, blueprint_library, spawn_points):
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
        if n >= 100:
            break
        blueprint = random.choice(vehicle_bp)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        # spawn
        print("spawned")

        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, False):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    return vehicles_list


def generate_walkers(client, world, blueprint_library, spawn_points, number_of_walkers):
    return 0


def filter_angle(vehicles_list, world, vehicle, fov):
    filtered_vehicles = []
    for npc in world.get_actors().filter('vehicle*'):
        if npc.id in vehicles_list and npc.id != vehicle.id:
            # npc transform
            npc_transform = npc.get_transform()
            # vehicle transform
            vehicle_transform = vehicle.get_transform()
            # angle between npc and vehicle
            angle = np.arctan2(npc_transform.location, vehicle_transform.location) * 180 / np.pi
            selector = np.array(np.absolute(angle) < (int(fov) / 2))
            if selector:
                filtered_vehicles.append(npc)
    return filtered_vehicles


def main(town):
    # Simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    #world = client.load_world(town)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    blueprint = blueprint_library.filter('model3')

    vehicle = world.spawn_actor(blueprint[0], random.choice(spawn_points))

    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
    vehicle.set_autopilot(True)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05  # (20fps)
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Spawn liDar
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('rotation_frequency', '20')
    # fov
    lidar_bp.set_attribute('points_per_second', '1300000')
    lidar_bp.set_attribute('upper_fov', str(7))
    lidar_bp.set_attribute('lower_fov', str(-16))
    # lidar_bp.set_attribute('horizontal_fov', str(360))
    lidar_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0.8, z=1.7)), attach_to=vehicle)
    # Create a queue to store and retrieve the sensor data
    lidar_queue = queue.Queue()
    lidar.listen(lidar_queue.put)

    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)


    # Get the bounding boxes from traffic lights used later for red light detection
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)


    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    vehicles_list = generate_traffic(traffic_manager, client, blueprint_library, spawn_points)

    world.tick()
    image = image_queue.get()
    pointcloud = lidar_queue.get()

    # Reshape the raw data into an RGB array
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # Reshape pointcloud data
    lidar_data = np.frombuffer(pointcloud.raw_data, dtype=np.dtype('f4'))

    # Display the image in an OpenCV display window
    cv2.namedWindow('CARLA RaceAI', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('CARLA RaceAI', img)
    cv2.waitKey(1)
    try:
        ### Game loop ###
        while True:
                # Retrieve and reshape the image
                world.tick()
                image = image_queue.get()
                pointcloud = lidar_queue.get()

                img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

                # Get the camera matrix
                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
                # only take measurements every 20 frames
                if image.frame % 20 == 0:
                    # Save the image -- for export
                    image_path = 'output/camera_output/images/%06d' % image.frame

                    image.save_to_disk(image_path + '.png')

                    # Initialize the exporter
                    writer = Writer(image_path + '.png', image_w, image_h)
                    boxes = []
                    filtered_list = filter_angle(vehicles_list, world, vehicle, fov)
                    for npc in filtered_list:
                        # Filter out the ego vehicle
                        if npc.id != vehicle.id and npc.id in vehicles_list:
                            #print(npc.id)
                            bb = npc.bounding_box
                            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                            # Filter for the vehicles within 50m
                            if 0.5 < dist < 50:
                                forward_vec = vehicle.get_transform().get_forward_vector()
                                ray = npc.get_transform().location - vehicle.get_transform().location

                                if forward_vec.dot(ray) > 1:
                                    p1 = get_image_point(bb.location, K, world_2_camera)
                                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                    x_max = -10000
                                    x_min = 10000
                                    y_max = -10000
                                    y_min = 10000

                                    for vert in verts:
                                        p = get_image_point(vert, K, world_2_camera)
                                        # Find the rightmost vertex
                                        if p[0] > x_max:
                                            x_max = p[0]
                                        # Find the leftmost vertex
                                        if p[0] < x_min:
                                            x_min = p[0]
                                        # Find the highest vertex
                                        if p[1] > y_max:
                                            y_max = p[1]
                                        # Find the lowest  vertex
                                        if p[1] < y_min:
                                            y_min = p[1]
                                    already_there = False
                                    for box in boxes:
                                        if box[0] <= x_min and box[1] <= y_min and box[2] >= x_max and box[3] >= y_max:
                                            already_there = True
                                    if not already_there:
                                        cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0,0,255, 255), 1)
                                        cv2.line(img, (int(x_min), int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                                        # get name of the vehicle
                                        name = npc.type_id.split('.')[2]
                                        print(name)

                                        classification = 'vehicle'
                                        if name == 'crossbike' or name == 'low_rider' or name == 'ninja' or name == 'zx125' or name == 'yzf' or name == 'omafiets':
                                            classification = 'motorcycle'
                                        elif name == 'firetruck' or name == 'ambulance' or name == 'sprinter' or name == 'carlacola':
                                            classification = 'truck'
                                        print(classification)
                                        # not already a bounding box there

                                        if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                                            writer.addObject(classification, x_min, y_min, x_max, y_max)
                                            boxes.append([x_min, y_min, x_max, y_max])



                    # Save the bounding boxes in the scene
                    writer.save(image_path + '.xml')

                    cv2.imshow('CARLA RaceAI', img)

                # Save liDAR data and create 3D bounding boxes
                if pointcloud.frame % 20 == 0:
                    # Save the pointcloud-- for export
                    #lidar_path = 'output/lidar_output/%06d' % pointcloud.frame
                    #pointcloud.save_to_disk(lidar_path + '.ply')
                    # I think maybe z should be -z coord?
                    # Save the 3D bounding boxes of all the cars in the scene in xml file
                    #lidar_data = np.frombuffer(pointcloud.raw_data, dtype=np.dtype('f4'))
                    #lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
                    #lidar_data = lidar_data[:, :3]
                    #lidar_data = lidar_data.reshape(-1)
                    #lidar_data = np.array(lidar_data, dtype=np.float32)
                    #lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 3), 3))



                    if cv2.waitKey(1) == ord('q'):
                        break
    finally:
        # Destroy the actors
        for actor in world.get_actors().filter('vehicle*'):
            actor.destroy()
        print('All actors destroyed.')

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main('Town10')
