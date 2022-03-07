import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math

def load_train_dataset(N=100,NP=4000,folder='../data/simple/',s=0):
	# load data as [path]
	# for each path, it is
	# [[input],[target],[env_id]]
	obs = []
	# add start s
	for i in range(0,N):
		#load obstacle point cloud
		temp=np.fromfile(folder+'obs_cloud/obc'+str(i+s)+'.dat')
		obs.append(temp)
	obs = np.array(obs)

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'e'+str(i+s)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'e'+str(i+s)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]


	dataset=[]
	targets=[]
	env_indices=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:
				for m in range(0, path_lengths[i][j]-1):
					data = np.concatenate( (paths[i][j][m], paths[i][j][path_lengths[i][j]-1]) ).astype(np.float32)
					targets.append(paths[i][j][m+1])
					dataset.append(data)
					env_indices.append(i)
	# only return raw data (in order), follow below to randomly shuffle
	data=list(zip(dataset,targets,env_indices))
	random.shuffle(data)
	dataset,targets,env_indices=list(zip(*data))
	dataset = list(dataset)
	targets = list(targets)
	env_indices = list(env_indices)
	return obs, dataset, targets, env_indices

def load_obs_list(env_id, folder='../data/simple'):
	# load shape representation of obstacle
	obc=np.zeros((7,2),dtype=np.float32)
	temp=np.fromfile(folder+'obs.dat')
	obs=temp.reshape(len(temp)//2,2)
	temp=np.fromfile(folder+'obs_perm2.dat',np.int32)
	perm=temp.reshape(77520,7)
	for j in range(0,7):
		print(obs[perm[env_id][j]])
		for k in range(0,2):
			obc[j][k]=obs[perm[env_id][j]][k]
	# returns a list of 2d obstacles expressed by their centers (x,y)
	return obc


#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100,NP=200, s=0,sp=4000, folder='../data/simple/',rand=False):
	# load shape representation of obstacle
	obc=np.zeros((N,7,2),dtype=np.float32)
	temp=np.fromfile(folder+'obs.dat')
	obs=temp.reshape(len(temp)//2,2)
	temp=np.fromfile(folder+'obs_perm2.dat',np.int32)
	perm=temp.reshape(77520,7)
	envs_array=range(0,N)
	paths_array=range(0,NP)
	if rand==True:
		if s==0:
			envs_array=np.random.randint(0,100,(N,))
		if sp==4000:
			paths_array=np.random.randint(0,100,(NP,))
	for i in range(0,N):
		for j in range(0,7):
			for k in range(0,2):
				obc[i][j][k]=obs[perm[envs_array[i]+s][j]][k]

	# load point cloud representation of obstacle
	obs = []
	for i in envs_array:
		temp=np.fromfile(folder+'obs_cloud/obc'+str(i+s)+'.dat')
		obs.append(temp)
	obs = np.array(obs).astype(np.float32)

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'e'+str(envs_array[i]+s)+'/path'+str(paths_array[j]+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'e'+str(envs_array[i]+s)+'/path'+str(paths_array[j]+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]

	return 	obc,obs,paths,path_lengths, envs_array, paths_array
