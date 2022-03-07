import torch
import numpy as np
from utility import *
import time
DEFAULT_STEP = 0.01
def removeCollision(path, obc, IsInCollision):
    new_path = []
    # rule out nodes that are already in collision
    for i in range(0,len(path)):
        if not IsInCollision(path[i].numpy(),obc):
            new_path.append(path[i])
    return new_path

def steerTo(start, end, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # test if there is a collision free path from start to end, with step size
    # given by step_sz, and with generic collision check function
    # here we assume start and end are tensors
    # return 0 if in coliision; 1 otherwise
    start_t = time.time()
    DISCRETIZATION_STEP=step_sz
    delta = end - start  # change
    delta = delta.numpy()
    total_dist = np.linalg.norm(delta)
    # obtain the number of segments (start to end-1)
    # the number of nodes including start and end is actually num_segs+1
    num_segs = int(total_dist / DISCRETIZATION_STEP)
    if num_segs == 0:
        # distance smaller than threshold, just return 1
        return 1
    # obtain the change for each segment
    delta_seg = delta / num_segs
    # initialize segment
    seg = start.numpy()
    # check for each segment, if they are in collision
    for i in range(num_segs+1):
        if IsInCollision(seg, obc):
            # in collision
            return 0
        seg = seg + delta_seg
    return 1

def feasibility_check(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # checks the feasibility of entire path including the path edges
    # by checking for each adjacent vertices
    for i in range(0,len(path)-1):
        if not steerTo(path[i],path[i+1],obc,IsInCollision,step_sz=step_sz):
            # collision occurs from adjacent vertices
            return 0
    return 1
def lvc(path, obc, IsInCollision, step_sz=DEFAULT_STEP):
    # lazy vertex contraction
    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            ind=0
            ind=steerTo(path[i],path[j],obc,IsInCollision,step_sz=step_sz)
            if ind==1:
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])
                return lvc(pc,obc,IsInCollision,step_sz=step_sz)
    return path
def neural_plan(mpNet, path, obc, obs, IsInCollision, normalize, unnormalize, init_plan_flag, step_sz=DEFAULT_STEP):
    if init_plan_flag:
        # if it is the initial plan, then we just plan from start to goal
        MAX_LENGTH = 80
        mini_path = neural_mini_planner(mpNet, path[0], path[-1], obc, obs, IsInCollision, \
                                            normalize, unnormalize, MAX_LENGTH, step_sz=step_sz)
        if mini_path:
            # if mini plan is successful
            return removeCollision(mini_path, obc, IsInCollision)
        else:
            # can't find a path
            return path
    MAX_LENGTH = 50
    # replan segments of paths
    new_path = [path[0]]
    time_norm = 0.
    for i in range(len(path)-1):
        # look at if adjacent nodes can be connected
        # assume start is already in new path
        start = path[i]
        goal = path[i+1]
        steer = steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)
        if steer:
            new_path.append(goal)
        else:
            # plan mini path
            mini_path = neural_mini_planner(mpNet, start, goal, obc, obs, IsInCollision, \
                                            normalize, unnormalize, MAX_LENGTH, step_sz=step_sz)
            if mini_path:
                # if mini plan is successful
                new_path += removeCollision(mini_path[1:], obc, IsInCollision)  # take out start point
            else:
                new_path += path[i+1:]     # just take in the rest of the path
                break
    return new_path


def neural_mini_planner(mpNet, start, goal, obc, obs, IsInCollision, normalize, unnormalize, MAX_LENGTH, step_sz=DEFAULT_STEP):
    # plan a mini path from start to goal
    # obs: tensor
    itr=0
    pA=[]
    pA.append(start)
    pB=[]
    pB.append(goal)
    target_reached=0
    tree=0
    new_path = []
    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long
        if tree==0:
            ip1 = torch.cat((start, goal)).unsqueeze(0)
            ob1 = torch.FloatTensor(obs).unsqueeze(0)
            #ip1=torch.cat((obs,start,goal)).unsqueeze(0)
            time0 = time.time()
            ip1=normalize(ip1)
            ip1=to_var(ip1)
            ob1=to_var(ob1)
            start=mpNet(ip1,ob1).squeeze(0)
            # unnormalize to world size
            start=start.data.cpu()
            time0 = time.time()
            start = unnormalize(start)
            pA.append(start)
            tree=1
        else:
            ip2 = torch.cat((goal, start)).unsqueeze(0)
            ob2 = torch.FloatTensor(obs).unsqueeze(0)
            #ip2=torch.cat((obs,goal,start)).unsqueeze(0)
            time0 = time.time()
            ip2=normalize(ip2)
            ip2=to_var(ip2)
            ob2=to_var(ob2)
            goal=mpNet(ip2,ob2).squeeze(0)
            # unnormalize to world size
            goal=goal.data.cpu()
            time0 = time.time()
            goal = unnormalize(goal)
            pB.append(goal)
            tree=0
        target_reached=steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)

    if target_reached==0:
        return 0
    else:
        for p1 in range(len(pA)):
            new_path.append(pA[p1])
        for p2 in range(len(pB)-1,-1,-1):
            new_path.append(pB[p2])

    return new_path
