import carla

def carla_API():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    #world = client.get_world()
    # Change state to synchronous mode when you want to control the simulation (when deploying the agent)
    # Load town 01 map
    world = client.load_world('Town01')


if __name__ == '__main__':
    carla_API()

