import carla

def carla_API():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    #world = client.get_world()
    # Change state to synchronous mode when you want to control the simulation (when deploying the agent)
    # Load town 01 map
    world = client.load_world('Town02')
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, carla.Transform(carla.Location(x=0, y=0, z=2), carla.Rotation(yaw=0)))


if __name__ == '__main__':
    carla_API()

