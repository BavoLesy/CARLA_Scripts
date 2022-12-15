import os

import numpy as np
import plyfile
import os
import shutil

def convert_2_bin(folder, k):
    #Convert every file in this folder
    for filename in os.listdir(folder):
        lidar_path = 'output/lidar_output/Town10HD/ply/' + filename
        plydata = plyfile.PlyData.read(lidar_path)
        lidar = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'], plydata['vertex']['I']]).transpose()
        # flip y axis
        lidar[:, 1] *= -1
        # save to bin file with intensity as color
        # start numbering at 000000.bin and increment
        # Make directory if it doesn't exist
        if not os.path.exists('output/lidar_output/bin'):
            os.makedirs('output/lidar_output/bin')
        lidar.tofile('output/lidar_output/bin/' + str(k).zfill(6) + '.bin')

        # Rename label filename to match bin files
        label_path = 'output/lidar_output/Town10HD/labels/' + filename[:-3] + 'txt'
        label = open(label_path, 'r')
        # save file under new name
        if not os.path.exists('output/lidar_output/labels'):
            os.makedirs('output/lidar_output/labels')
        new_label = open('output/lidar_output/labels/' + str(k).zfill(6) + '.txt', 'w')
        new_label.write(label.read())
        k += 1
    return k



def parse_over_files(directory, i, j):
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):

            newname = str(i).zfill(6) + ".xml"
            # if not exist
            i += 1
            if not os.path.exists('output/camera_output/annotations'):
                os.makedirs('output/camera_output/annotations')
            target = r'output/camera_output/annotations/' + newname
            shutil.copyfile(os.path.join(directory, filename), target)

        if filename.endswith(".png"):

            newname = str(j).zfill(6) + ".png"
            # if not exist
            if not os.path.exists('output/camera_output/images'):
                os.makedirs('output/camera_output/images')
            target = r'output/camera_output/images/' + newname
            shutil.copyfile(os.path.join(directory, filename), target)
            j += 1

    return i, j






if __name__ == '__main__':
    k = 0
    k = convert_2_bin('output/lidar_output/Town10HD/ply/', k)

    i = 0
    j = 0
    #i, j = parse_over_files('output/camera_output/Town01', i, j)
    #i, j = parse_over_files('output/camera_output/Town02', i, j)
    #i, j = parse_over_files('output/camera_output/Town03', i, j)
    #i, j = parse_over_files('output/camera_output/Town04', i, j)
    #i, j = parse_over_files('output/camera_output/Town05', i, j)
    i, j = parse_over_files('output/camera_output/Town10HD', i, j)
