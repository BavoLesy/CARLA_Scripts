
"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""

import logging
import os
from math import cos, sin
import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import plyfile
from pascal_voc_writer import Writer
from CARLA_utils.ply2bin import ply2bin

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
        # print("spawned")

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


def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


### Get numpy 2D array of vehicles' location and rotation from world reference, also locations from sensor reference
def get_list_transform(vehicles_list, sensor):
    t_list = []
    for vehicle in vehicles_list:
        v = vehicle.get_transform()
        transform = [v.location.x, v.location.y, v.location.z, v.rotation.roll, v.rotation.pitch, v.rotation.yaw]
        t_list.append(transform)
    t_list = np.array(t_list).reshape((len(t_list), 6))

    transform_h = np.concatenate((t_list[:, :3], np.ones((len(t_list), 1))), axis=1)
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    transform_s = np.dot(world_sensor_matrix, transform_h.T).T

    return t_list, transform_s


def filter_angle_occlusion(vehicles_list, world, vehicle, fov):
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


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def main(town, num_of_vehicles, num_of_walkers, num_of_frames):
    # Simulator
    global semantic_list
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.load_world(town)
    # world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    blueprint = blueprint_library.filter('model3')

    ego = world.spawn_actor(blueprint[0], random.choice(spawn_points))

    # spawn camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego)
    ego.set_autopilot(True)

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
    lidar_bp.set_attribute('range', '80')
    lidar_bp.set_attribute('rotation_frequency', '20')
    # fov
    lidar_bp.set_attribute('points_per_second', str(64 / 0.00004608))
    lidar_bp.set_attribute('upper_fov', str(2))
    lidar_bp.set_attribute('lower_fov', str(-24.8))
    # lidar_bp.set_attribute('horizontal_fov', str(360))
    # lidar_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0)), attach_to=ego)
    # Create a queue to store and retrieve the sensor data
    lidar_queue = queue.Queue()
    lidar.listen(lidar_queue.put)
    #Semantic lidar
    sem_lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    sem_lidar_bp.set_attribute('channels', '32')
    sem_lidar_bp.set_attribute('range', '80')
    sem_lidar_bp.set_attribute('rotation_frequency', '20')
    # fov
    sem_lidar_bp.set_attribute('points_per_second', str(64 / 0.00004608 * 2))
    sem_lidar_bp.set_attribute('upper_fov', str(2))
    sem_lidar_bp.set_attribute('lower_fov', str(-24.8))

    sem_lidar = world.spawn_actor(sem_lidar_bp, carla.Transform(carla.Location(x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=0, roll=0)), attach_to=ego)
    sem_lidar_queue = queue.Queue()
    sem_lidar.listen(sem_lidar_queue.put)
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
    vehicles_list = generate_traffic(traffic_manager, client, blueprint_library, spawn_points, num_of_vehicles)
    # Spawn pedestrians and also detect the bounding boxes

    # Detect traffic Lights bounding boxes

    world.tick()
    image = image_queue.get()
    pointcloud = lidar_queue.get()

    # Reshape the raw data into an RGB array
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # Reshape pointcloud data
    #lidar_data = np.frombuffer(pointcloud.raw_data, dtype=np.dtype('f4'))

    # Display the image in an OpenCV display window
    cv2.namedWindow('CARLA RaceAI', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('CARLA RaceAI', img)
    cv2.waitKey(1)
    i = 50
    boxes = []
    try:
        ### Game loop ###
        while image.frame < num_of_frames:
            # Retrieve and reshape the image
            world.tick()
            image = image_queue.get()
            pointcloud = lidar_queue.get()
            sem_pointcloud = sem_lidar_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # Get the camera matrix
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            # only take measurements every 50 frames
            if image.frame % 30 == 0:

                i = 0
                # Save the image -- for export
                # Initialize the exporter

                boxes = []
                sem_lidar_data = np.frombuffer(sem_pointcloud.raw_data, dtype=np.dtype(
                    [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('cos', 'f4'), ('index', 'u4'), ('semantic', 'u4')]))
                points = np.array([sem_lidar_data[:]['index'], sem_lidar_data[:]['semantic']])
                mask = np.array([sem_lidar_data[:]['index'], sem_lidar_data[:]['semantic']])[:][1] == 10
                semantic_list = np.unique(points[0][mask])
                for npc in world.get_actors():
                    # Filter out the ego vehicle
                    if npc.id != ego.id and npc.id in vehicles_list and npc.id in semantic_list:
                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(ego.get_transform().location)

                        # Filter for the vehicles within 50m
                        if 0.5 < dist < 60:
                            forward_vec = ego.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - ego.get_transform().location

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
                                    # Find the lowest vertex
                                    if p[1] < y_min:
                                        y_min = p[1]
                                name = npc.type_id.split('.')[2]
                                classification = 'car'
                                if name == 'ambulance' or name == 'firetruck' or name == 'charger_police' or name == 'charger_police_2020':
                                    classification = 'emergency'
                                elif name == 'crossbike' or name == 'low_rider' or name == 'ninja' or name == 'zx125' or name == 'yzf':
                                    classification = 'motorcycle'
                                elif name == 'omafiets':
                                    classification = 'bicycle'
                                elif name == 'sprinter' or name == 'carlacola':
                                    classification = 'van'
                                    # Add the object to the frame (ensure it is inside the image)
                                x_min = np.clip(x_min, 0, image_w)
                                x_max = np.clip(x_max, 0, image_w)
                                y_min = np.clip(y_min, 0, image_h)
                                y_max = np.clip(y_max, 0, image_h)
                                if x_min != x_max and y_min != y_max:
                                    boxes.append([x_min, y_min, x_max, y_max, classification])
            i += 1
            if i == 3:
                # Compare the bounding boxes to every other bounding box
                # Filter out bad boxes
                for box in boxes:
                    for other_box in boxes:
                        # If the boxes are the same, skip
                        if box != other_box:
                            box_w = box[2] - box[0]
                            box_h = box[3] - box[1]
                            other_box_w = other_box[2] - other_box[0]
                            other_box_h = other_box[3] - other_box[1]
                            # Check if box is fully contained in other_box
                            if other_box[0] <= box[0] and other_box[1] <= box[1] and other_box[2] >= box[2] and \
                                    other_box[3] >= box[3]:
                                # If the box is fully contained, remove it
                                boxes.remove(box)
                                break
                        """
                            # If the box is 80% contained in other_box, remove it
                            elif (other_box[0] <= box[0] and other_box[2] <= (box[2]+(box_w*0.2))) and (other_box[1] <= box[1] and other_box[3] <= (box[3]+(box_h*0.2))):
                                boxes.remove(box)
                                break
                            elif (other_box[0] >= (box[0]-(box_w*0.2)) and other_box[2] >= box[2]) and (other_box[1] <= box[1] and other_box[3] <= (box[3]+(box_h*0.2))):
                                boxes.remove(box)
                                break
                            elif (other_box[0] <= box[0] and other_box[2] <= (box[2]+(box_w*0.2))) and (other_box[1] >= (box[1]-(box_h*0.2)) and other_box[3] >= box[3]):
                                boxes.remove(box)
                                break
                            elif (other_box[0] >= (box[0]-(box_w*0.2)) and other_box[2] >= box[2]) and (other_box[1] >= (box[1]-(box_h*0.2)) and other_box[3] >= box[3]):
                                boxes.remove(box)
                                break
                            elif (other_box[0] <= box[0]+(box_w*0.2) and other_box[2] >= box[2]-(box_w*0.2)) and (other_box[1] <= box[1] and other_box[3] >= box[3]):
                                boxes.remove(box)
                                break
                            elif (other_box[0] <= box[0] and other_box[2] >= box[2]) and (other_box[1] <= box[1]+(box_h*0.2) and other_box[3] >= box[3]-(box_h*0.2)):
                                boxes.remove(box)
                                break







                            # Check if box is contained in multiple other_boxes
                            if other_box[0] <= box[0]:
                                xmin_bool = True
                            if other_box[1] <= box[1]:
                                ymin_bool = True
                            if other_box[2] >= box[2]:
                                xmax_bool = True
                            if other_box[3] >= box[3]:
                                ymax_bool = True
                    if xmin_bool and ymin_bool and xmax_bool and ymax_bool:
                        boxes.remove(box)
                    """
                image_path = 'output/camera_output/' + town + '/' + '%06d' % image.frame
                image.save_to_disk(image_path + '.png')
                writer = Writer(image_path + '.png', image_w, image_h)
                for box in boxes:
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), (0, 0, 255, 255), 1)
                    cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255, 255), 1)

                    writer.addObject(box[4], box[0], box[1], box[2], box[3])

                    # Save the bounding boxes in the scene
                writer.save(image_path + '.xml')

                cv2.imshow('CARLA RaceAI', img)
                # save the image
                if not os.path.exists('output/camera_output/' + town + '/bbox'):
                    os.makedirs('output/camera_output/' + town + '/bbox/')
                cv2.imwrite('output/camera_output/' + town + '/bbox/' + str(image.frame) + '.png', img)

            # Save liDAR data and create 3D bounding boxes
            if (pointcloud.frame % 30) - 2 == 0:
                # get location from the lidar sensor
                lidar_location = lidar.get_transform().location
                lidar_transform = lidar.get_transform()
                inv_transform = carla.Transform(carla.Location(0, 0, 0),
                                                carla.Rotation(-lidar_transform.rotation.pitch, -lidar_transform.rotation.yaw,
                                                               -lidar_transform.rotation.roll))
                labels = []
                for npc in world.get_actors():
                    # Filter out the ego vehicle
                    if npc.id != ego.id and npc.id in vehicles_list and npc.id in semantic_list:
                        transform = npc.get_transform()
                        if lidar_location.distance(transform.location) < 50:
                            # Get the bounding box of the vehicle
                            bounding_box = npc.bounding_box
                            # Get the corners of the bounding box
                            corners = bounding_box.extent
                            # Get the rotation of the vehicle
                            rotation = transform.rotation
                            # get the location of the vehicle
                            location = bounding_box.location

                            point = inv_transform.transform(transform.location - lidar_location)



                            # get type of the vehicle
                            name = npc.type_id.split('.')[2]
                            classification = 'car'
                            if name == 'ambulance' or name == 'firetruck' or name == 'charger_police' or name == 'charger_police_2020':
                                classification = 'emergency'
                            elif name == 'crossbike' or name == 'low_rider' or name == 'ninja' or name == 'zx125' or name == 'yzf':
                                classification = 'motorcycle'
                            elif name == 'omafiets':
                                classification = 'bicycle'
                            elif name == 'sprinter' or name == 'carlacola':
                                classification = 'van'
                            labels.append({
                                'type': "DontCare",
                                'truncated': 0,
                                'occluded': 0,
                                'alpha': 0,
                                'xmin': 0,
                                'ymin': 0,
                                'xmax': 0,
                                'ymax': 0,
                                'height': round(corners.z * 2, 2),
                                'width': round(corners.y * 2, 2),
                                'length': round(corners.x * 2, 2),
                                'x': round(location.x + point.x, 2),
                                'y': round(location.y + point.y, 2),
                                'z': round(location.z + point.z, 2),
                                'yaw': np.radians(rotation.yaw - lidar_transform.rotation.yaw),

                            })
                #make directory for the pointclouds
                if not os.path.exists('output/lidar_output/' + town + '/labels'):
                    os.makedirs('output/lidar_output/' + town + '/labels')
                if not os.path.exists('output/lidar_output/' + town + '/data'):
                    os.makedirs('output/lidar_output/' + town + '/data')
                if not os.path.exists('output/lidar_output/' + town + '/ply'):
                    os.makedirs('output/lidar_output/' + town + '/ply')
                with open('output/lidar_output/' + town + '/labels/' + '%06d' % pointcloud.frame + '.txt', 'w') as f:
                    for label in labels:
                        f.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % (
                            label['type'], label['truncated'], label['occluded'], label['alpha'], label['xmin'],
                            label['ymin'], label['xmax'], label['ymax'], label['height'], label['width'], label['length'],
                            label['x'], label['y'], label['z'], label['yaw']))

                lidar_path = 'output/lidar_output/' + town + '/ply/' + '%06d' % pointcloud.frame + '.ply'
                pointcloud.save_to_disk(lidar_path)
                # flip the pointcloud y axis
                """
                plydata = plyfile.PlyData.read(lidar_path)
                lidar = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'],
                                   plydata['vertex']['I']]).transpose()
                # flip y axis
                lidar[:, 1] *= -1
                # save to bin file with intensity as color
                lidar.tofile('output/lidar_output/' + town + '/data/' + '%06d' % pointcloud.frame + '.bin')
                """
                if cv2.waitKey(1) == ord('q'):
                    break
    finally:
        # Destroy the actors
        for actor in world.get_actors().filter('vehicle.*'):
            actor.destroy()
        for actor in world.get_actors().filter('sensor.*'):
            actor.destroy()
        print('All actors destroyed.')

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Measurement every 50 frames, we want 400 measurement per town, so 30 * 400 = 16000+ 4000 for bad measurements
    # TO DO: change weather dynamically for each town

    frames = 20000
    num_vehicle = 75
    num_pedestrian = 30
    #main('Town04', num_vehicle, num_pedestrian, frames)
    main('Town10HD', num_vehicle, num_pedestrian, frames)
    main('Town01', num_vehicle, num_pedestrian, frames)
    main('Town02', num_vehicle, num_pedestrian, frames)
    main('Town03', num_vehicle, num_pedestrian, frames)
    main('Town05', num_vehicle, num_pedestrian, frames)



