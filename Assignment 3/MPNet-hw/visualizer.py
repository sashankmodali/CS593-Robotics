import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct
import numpy as np
import argparse
import os

import data_loader_2d

def main(args):
    if args.point_cloud:
        # visualize obstacles as point cloud
        file = args.data_path + f'obs_cloud/obc{args.env_id}.dat'
        obs = []
        temp=np.fromfile(file)
        obs.append(temp)
        obs = np.array(obs).astype(np.float32).reshape(-1,2)
        plt.scatter(obs[:,0], obs[:,1], c='gray')
    else:
        # visualize obstacles as rectangles
        size = 5
        obc = data_loader_2d.load_obs_list(args.env_id, folder=args.data_path)
        for (x,y) in obc:
            r = mpatches.Rectangle((x-size/2,y-size/2),size,size,fill=True,color='gray')
            plt.gca().add_patch(r)
    
    for path_file in args.path_file:
        # visualize path
        if path_file.endswith('.txt'):
            # print(path_file)
            path = np.loadtxt(path_file)
        else:
            # print(path_file)
            path = np.fromfile(path_file)
        #   print(path)
        path = path.reshape(-1, 2)
        path_x = []
        path_y = []
        for i in range(len(path)):
            path_x.append(path[i][0])
            path_y.append(path[i][1])
        if "data/" in path_file:
            plt.plot(path_x, path_y, '--r', label="RRT*", marker='o') 
        else:
            plt.plot(path_x, path_y, '-b', label="MPNet-run-{}".format(path_file[-5]), marker='o')

    if args.vis_dir!="":
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.title("Comparing MPNet paths for env {}".format(args.env_id))
        if not os.path.isdir(args.vis_dir):
            os.makedirs(args.vis_dir)
        plt.savefig(args.vis_dir + "env_{}".format(args.env_id) + "-comparision.png",bbox_inches='tight')

    if not args.blind:
        plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='../data/simple/')
parser.add_argument('--env-id', type=int, default=0)
parser.add_argument('--point-cloud', default=False, action='store_true')
parser.add_argument('--blind', default=False, action='store_true')
parser.add_argument('--vis-dir', type=str, default='', help='directory for saving visualization. If empty, not saved')
parser.add_argument('--path-file', nargs='*', type=str, default=[], help='path file')

args = parser.parse_args()
print(args)
main(args)
