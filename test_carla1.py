import argparse
import logging
import random

import carla
import pygame as pygame


def carla_API():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    # Change state to synchronous mode when you want to control the simulation (when deploying the agent)
    # Load town 01 map
    # world = client.load_world('Town02')
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    actor_list.append(vehicle)
    print('created %s' % vehicle.type_id)
    vehicle.set_autopilot(True)
    # We need sensors LIDAR on the vehicle to get the data
    # We need to spawn the vehicle and attach the sensors to it
    # attach sensors to the vehicle
    # LIDAR with 64 channels, 1.3 M points per second, 120m range, -25 to 5 vertical FOV, +- 2cm error
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '120')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('points_per_second', '1300000')
    lidar_bp.set_attribute('upper_fov', '5')
    lidar_bp.set_attribute('lower_fov', '-25')
    lidar_bp.set_attribute('sensor_tick', '0.05')
    lidar_bp.set_attribute('dropoff_general_rate', '1.0')
    lidar_bp.set_attribute('dropoff_intensity_limit', '0.0')
    lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    # spawn the sensor and attach to vehicle.
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    lidar = world.spawn_actor(lidar_bp, spawn_point, attach_to=vehicle)
    actor_list.append(lidar)
    print('created %s' % lidar.type_id)
    # Get outputs of the lidar sensor into measurement files
    lidar.listen(lambda data: data.save_to_disk('output/%06d.bin' % data.frame))
    # We need to get the data from the lidar sensor
    return client





def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()


    try:
        if args.seed:
            random.seed(args.seed)

        client = carla_API()

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)



        clock = pygame.time.Clock()


        while True:
            clock.tick()
            if args.sync:
                sim_world.tick()
            else:
                sim_world.wait_for_tick()

            sim_world.tick(clock)
            sim_world.render(display)
            pygame.display.flip()



    finally:

        if sim_world is not None:
            settings = sim_world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            #sim_world.destroy()

        pygame.quit()




def main():
    """Main method"""
    argparser = argparse.ArgumentParser(
        description='CARLA Testing Sensors data Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Basic")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()


