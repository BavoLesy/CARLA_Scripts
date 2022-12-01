import carla

def carla_API():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    #world = client.get_world()
    # Change state to synchronous mode when you want to control the simulation (when deploying the agent)
    # Load town 01 map
    world = client.load_world('Town01')

    """
    settings = world.get_settings()
    settings.syncronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    """
    # We need sensors (LIDAR + 2 CAMERA) on the vehicle to get the data
    # We need to spawn the vehicle and attach the sensors to it

    # Spawn the vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, carla.Transform(carla.Location(x=0, y=0, z=2), carla.Rotation(yaw=0)))
    # Attach the sensors to the vehicle
    # LIDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '5000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('points_per_second', '1000000')
    lidar_bp.set_attribute('upper_fov', '10')
    lidar_bp.set_attribute('lower_fov', '-30')
    lidar_bp.set_attribute('sensor_tick', '0.05')
    lidar_bp.set_attribute('dropoff_general_rate', '1.0')
    lidar_bp.set_attribute('dropoff_intensity_limit', '0.0')
    lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    lidar_bp.set_attribute('noise_stddev', '0.0')
    lidar_bp.set_attribute('enable_noise', 'false')
    lidar_bp.set_attribute('horizontal_fov', '100')
    lidar_bp.set_attribute('vertical_fov', '50')
    lidar_bp.set_attribute('sensor_tick', '0.05')
    # Attach LIDAR
    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2), carla.Rotation(yaw=0))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    # Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    # Set the time in seconds between sensor captures
    camera_bp.set_attribute('sensor_tick', '0.05')
    # Attach Camera
    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=2), carla.Rotation(yaw=0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    # Camera 2
    camera2_bp = blueprint_library.find('sensor.camera.rgb')
    camera2_bp.set_attribute('image_size_x', '800')
    camera2_bp.set_attribute('image_size_y', '600')
    camera2_bp.set_attribute('fov', '90')
    camera2_bp.set_attribute('sensor_tick', '0.05')
    # Attach Camera 2
    camera2_transform = carla.Transform(carla.Location(x=0, y=0, z=2), carla.Rotation(yaw=0))
    camera2 = world.spawn_actor(camera2_bp, camera2_transform, attach_to=vehicle)
    # Get the data from the sensors
    # LIDAR
    lidar.listen(lambda data: data.save_to_disk('output/lidar/%06d.ply' % data.frame))
    # Camera
    camera.listen(lambda data: data.save_to_disk('output/camera/%06d.png' % data.frame))
    # Camera 2
    camera2.listen(lambda data: data.save_to_disk('output/camera2/%06d.png' % data.frame))
    # Input data in



    print(world)




if __name__ == '__main__':
    carla_API()

