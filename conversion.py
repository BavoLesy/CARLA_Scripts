import os

import numpy as np
import plyfile
def main(filename):
    #Convert every file in this folder
     for filename in os.listdir(folder):
        lidar_path = 'output/lidar_output/Town10HD/ply/' + filename + '.ply'
    plydata = plyfile.PlyData.read(lidar_path)
    lidar = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'], plydata['vertex']['I']]).transpose()
    # flip y axis
    lidar[:, 1] *= -1
    # save to bin file with intensity as color


    lidar.tofile(lidar_path[:-3] + 'bin')

if __name__ == '__main__':
    folder = 'output/lidar_output/Town10HD/ply/'
    main(folder)
    main('000032')
    main('000062')
    main('000092')
    main('000122')
    main('000152')
    main('000182')
