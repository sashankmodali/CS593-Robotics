import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import time

import struct
import numpy as np
import argparse
import os

import data_loader_2d
import data_loader_r3d

def cuboid_data(o,size=(1,1,1)):

    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plot_cube(pos=(0,0,0), size=(1,1,1), ax=None, **kwargs):
    if ax !=None:
        X, Y, Z = cuboid_data( pos, size )
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)

def main(args):
    if args.point_cloud:
        # visualize obstacles as point cloud
        file = args.data_path + f'obs_cloud/obc{args.env_id}.dat'
        obs = []
        temp=np.fromfile(file)
        obs.append(temp)
        if args.env_type =="s2d":
            obs = np.array(obs).astype(np.float32).reshape(-1,2)
            plt.scatter(obs[:,0], obs[:,1], c='gray')
        elif args.env_type=="r3d":
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            obs = np.array(obs).astype(np.float32).reshape(-1,3)
            ax.scatter(obs[:,0], obs[:,1], obs[:,2], c='gray')
    else:
        # visualize obstacles as rectangles
        if args.env_type =="s2d":
            size = 5
            obc = data_loader_2d.load_obs_list(args.env_id, folder=args.data_path)
            for (x,y) in obc:
                r = mpatches.Rectangle((x-size/2,y-size/2),size,size,fill=True,color='gray')
                plt.gca().add_patch(r)
        elif args.env_type=="r3d":
            sizes = [(5,5,10),(5,10,5),(5,10,10), (10,5,5), (10,5,10), (10,10,5), (10,10,10), (5,5,5), (10,10,10), (5,5,5)]
            obc = data_loader_r3d.load_obs_list(args.env_id, folder=args.data_path)
            
            colors = ["gray" for i in range(10)];

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # ax.set_aspect('equal')

            for p,s,c in zip(obc,sizes,colors):
                plot_cube(pos=p, size=s, ax=ax, color=c,zorder=-1000,alpha=0.5)
    endpoints=[]
    startpoints=[]
    for path_file in args.path_file:
        # visualize path
        if path_file.endswith('.txt'):
            # print(path_file)
            path = np.loadtxt(path_file)
        else:
            # print(path_file)
            path = np.fromfile(path_file)
        #   print(path)
        if args.env_type=="s2d":
            path = path.reshape(-1, 2)
            path_x = []
            path_y = []
            for i in range(len(path)):
                path_x.append(path[i][0])
                path_y.append(path[i][1])
            if "data/" in path_file:
                plt.plot(path_x, path_y, '--r', label="RRT*-cost-{}".format(np.round(np.sum(np.linalg.norm(path[1:]-path[0:-1],axis=1)),2)), marker='o') 
            else:
                plt.plot(path_x, path_y, '-b', label="MPNet-run-{}-cost-{}".format(path_file[-5],np.round(np.sum(np.linalg.norm(path[1:]-path[0:-1],axis=1)),2)), marker='o')
        elif args.env_type=="r3d":
            plt.ion()
            path = path.reshape(-1, 3)
            endpoints.append(path[-1,:])
            startpoints.append(path[0,:])
            path_x = []
            path_y = []
            path_z = []
            for i in range(len(path)):
                path_x.append(path[i][0])
                path_y.append(path[i][1])
                path_z.append(path[i][2])
            ax = fig.gca(projection='3d')
            if "data/" in path_file:
                # print(path.shape)
                ax.plot3D(path_x, path_y, path_z, '--r', label="RRT*-cost-{}".format(np.round(np.sum(np.linalg.norm(path[1:,:]-path[0:-1,:],axis=1)),2)), marker='o',zorder=5) 
            else:
                # print(path.shape)
                ax.plot3D(path_x, path_y, path_z, '-b', label="MPNet-run-{}-cost-{}".format(path_file[-5],np.round(np.sum(np.linalg.norm(path[1:,0:]-path[0:-1,0:],axis=1)),2)), marker='o',zorder=5)

    if args.vis_dir!="":
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.title("Comparing MPNet paths for env {}".format(args.env_id))
        if not os.path.isdir(args.vis_dir + "{}/".format(args.env_type)):
            os.makedirs(args.vis_dir+ "{}/".format(args.env_type))
        endpoints = np.array(endpoints)
        startpoints = np.array(startpoints)
        if args.env_type=="s2d":
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
        elif args.env_type=="r3d":
            pitches=[];
            azimuths=[];
            for i in range(len(endpoints)):
                vec = endpoints[i,:] - startpoints[i,:]
                pitches.append(np.arcsin(np.dot(vec/np.linalg.norm(vec),[0,0,1])))
                azimuths.append(np.arccos(np.dot(vec[0:2]/np.linalg.norm(vec[0:2]),[1,0])))
            ax = fig.gca(projection='3d')
            ax.view_init(0*180/np.pi*np.mean(pitches),180/np.pi*np.mean(azimuths)+30)
            ax.set_xlabel("x-axis")
            ax.set_ylabel("y-axis")
            ax.set_zlabel("z-axis")
            plt.show(block=False)
            plt.pause(10)
        plt.tight_layout()
        plt.savefig(args.vis_dir + "{}/".format(args.env_type) + "env_{}".format(args.env_id) + "-comparision.png",bbox_inches='tight')
        # if args.env_type=="r3d":
        #     ax.view_init(0*180/np.pi*np.mean(pitches),180/np.pi*np.mean(azimuths)-30)
        #     plt.savefig(args.vis_dir + "{}/".format(args.env_type) + "env_{}".format(args.env_id) + "-comparision_2.png",bbox_inches='tight')

    if not args.blind:
        plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='../data/simple/')
parser.add_argument('--env-id', type=int, default=0)
parser.add_argument('--point-cloud', default=False, action='store_true')
parser.add_argument('--blind', default=False, action='store_true')
parser.add_argument('--vis-dir', type=str, default='', help='directory for saving visualization. If empty, not saved')
parser.add_argument('--env-type', type=str, default='s2d', help='environment type. s2d for simple 2d. r3d for complex 3d')
parser.add_argument('--path-file', nargs='*', type=str, default=[], help='path file')


args = parser.parse_args()
print(args)
main(args)
