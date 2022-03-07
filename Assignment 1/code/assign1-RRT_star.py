"""
Path Planning Sample Code with RRT*

author: Sashank Modali, code adapted from Ahmed Qureshi, which was adapted from AtsushiSakai(@Atsushi_twi)

"""


import argparse
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time


def diff(v1, v2):
    """
    Computes the difference v1 - v2, assuming v1 and v2 are both vectors
    """
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2,angle1=0,angle2=0):
    """
    Computes the Euclidean distance (L2 norm) between two points p1 and p2
    """
    return magnitude(diff(p1, p2)) + abs(angle1-angle2)

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea, alg, geom, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=100):
        """
        Sets algorithm parameters

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,width,height],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start)
        self.end = Node(goal)
        self.obstacleList = obstacleList
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.alg = alg
        self.geom = geom
        self.dof = dof

        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter

        self.goalfound = False
        self.solutionSet = set()

    def planning(self, animation=False):
        """
        Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
        You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        for i in range(self.maxIter):
            rnd = self.generatesample()
            nind = self.GetNearestListIndex(self.nodeList, rnd)


            rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])


            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                if self.alg == 'rrtstar':
                    nearinds = self.find_near_nodes(newNode) # you'll implement this method
                    newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                else:
                    newParent = None

                # insert newNode into the tree
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = dist(newNode.state, self.nodeList[newParent].state,newNode.angle,self.nodeList[newParent].angle) + self.nodeList[newParent].cost
                else:
                    pass # nind is already set as newNode's parent
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                if self.alg == 'rrtstar':
                    self.rewire(newNode, newNodeIndex, nearinds) # you'll implement this method

                if self.is_near_goal(newNode):
                    self.solutionSet.add(newNodeIndex)
                    self.goalfound = True

                if animation:
                    self.draw_graph(rnd.state,rnd.angle)

        return self.get_path_to_goal()

    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """
        # your code here

        mincost=None
        newparentind = None

        for i in nearinds:
            (valid,cost) = self.steerTo(newNode,self.nodeList[i])
            if valid:
                newparentind = i
                mincost =  self.nodeList[i].cost + cost
                break
            elif i==nearinds[-1]:
                mincost = None
                newparentind = None

        if newparentind is not None:
            for i in nearinds:
                (valid,cost) = self.steerTo(newNode,self.nodeList[i])
                if valid and self.nodeList[i].cost + cost < mincost :
                    mincost = self.nodeList[i].cost + cost
                    newparentind = i

        return newparentind

    def steerTo(self, dest, source):
        """
        Charts a route from source to dest, and checks whether the route is collision-free.
        Discretizes the route into small steps, and checks for a collision at each step.

        This function is used in planning() to filter out invalid random samples. You may also find it useful
        for implementing the functions in question 1.

        dest: destination node
        source: source node

        returns: (success, cost) tuple
            - success is True if the route is collision free; False otherwise.
            - cost is the distance from source to dest, if the route is collision free; or None otherwise.
        """

        newNode = copy.deepcopy(source)

        DISCRETIZATION_STEP=self.expandDis

        dists = np.zeros(self.dof, dtype=np.float32)
        for j in range(0,self.dof):
            dists[j] = dest.state[j] - source.state[j]

        distTotal = magnitude(dists)

        angledistTotal = dest.angle - source.angle 


        if distTotal>0:
            incrementTotal = distTotal/DISCRETIZATION_STEP

            for j in range(0,self.dof):
                dists[j] =dists[j]/incrementTotal

            numSegments = int(math.floor(incrementTotal))+1
            angle_increment = (angledistTotal)/numSegments

            stateCurr = np.zeros(self.dof,dtype=np.float32)
            for j in range(0,self.dof):
                stateCurr[j] = newNode.state[j]

            stateCurr = Node(stateCurr, newNode.angle)

            for i in range(0,numSegments):

                if not self.__CollisionCheck(stateCurr):
                    return (False, None)

                for j in range(0,self.dof):
                    stateCurr.state[j] += dists[j]
                stateCurr.angle += angle_increment

            if not self.__CollisionCheck(dest):
                return (False, None)

            return (True, distTotal)
        else:
            return (False, None)

    def generatesample(self):
        """
        Randomly generates a sample, to be used as a new node.
        This sample may be invalid - if so, call generatesample() again.

        You will need to modify this function for question 3 (if self.geom == 'rectangle')

        returns: random c-space vector
        """
        rnd_angle=0;
        if random.randint(0, 100) > self.goalSampleRate:
            sample=[]
            for j in range(0,self.dof):
                sample.append(random.uniform(self.minrand, self.maxrand))
            if self.geom=='rectangle':
                rnd_angle = random.uniform(0,2*np.pi)
            rnd=Node(sample,rnd_angle)
        else:
            rnd = self.end
        return rnd

    def is_near_goal(self, node):
        """
        node: the location to check

        Returns: True if node is within 5 units of the goal state; False otherwise
        """
        d = dist(node.state, self.end.state)
        if d < 5.0:
            return True
        return False

    @staticmethod
    def get_path_len(path):
        """
        path: a list of coordinates

        Returns: total length of the path
        """
        pathLen = 0
        for i in range(1, len(path)):
            pathLen += dist(path[i][0], path[i-1][0],path[i][1],path[i-1][1])

        return pathLen


    def gen_final_course(self, goalind):
        """
        Traverses up the tree to find the path from start to goal

        goalind: index of the goal node

        Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
        """
        path = [(self.end.state,self.end.angle)]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append((node.state,node.angle))
            goalind = node.parent
        path.append((self.start.state,self.start.angle))
        return path

    def find_near_nodes(self, newNode):
        """
        Finds all nodes in the tree that are "near" newNode.
        See the assignment handout for the equation defining the cutoff point (what it means to be "near" newNode)

        newNode: the node to be inserted.

        Returns: a list of indices of nearby nodes.
        """
        # Use this value of gamma
        GAMMA = 50

        # your code here


        nearinds = []
        listlen = len(self.nodeList)

        for i in range(listlen):
            if dist(self.nodeList[i].state,newNode.state,self.nodeList[i].angle,newNode.angle)<= GAMMA*(np.log(listlen)/listlen)**(1/(self.dof+int(self.geom=='rectangle'))): # check d(x, v) ≤ γ(logi/i) 1/n
                nearinds.append(i); # add near_node to list

        return nearinds # return list

    def rewire(self, newNode, newNodeIndex, nearinds):
        """
        Should examine all nodes near newNode, and decide whether to "rewire" them to go through newNode.
        Recall that a node should be rewired if doing so would reduce its cost.

        newNode: the node that was just inserted
        newNodeIndex: the index of newNode
        nearinds: list of indices of nodes near newNode
        """
        # your code here

        for i in nearinds:
            (valid,cost) = self.steerTo(self.nodeList[i],newNode)
            cost_temp = self.nodeList[i].cost
            if valid and cost + newNode.cost < cost_temp:
                self.nodeList[i].parent = newNodeIndex
                self.nodeList[i].cost = cost + newNode.cost
                for j in (self.nodeList[i].children):
                    self.nodeList[j].cost = self.nodeList[j].cost - cost_temp + self.nodeList[i].cost;
        pass

    def GetNearestListIndex(self, nodeList, rnd):
        """
        Searches nodeList for the closest vertex to rnd

        nodeList: list of all nodes currently in the tree
        rnd: node to be added (not currently in the tree)

        Returns: index of nearest node
        """
        dlist = []
        for node in nodeList:
            dlist.append(dist(rnd.state, node.state,rnd.angle,node.angle))

        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node):
        """
        Checks whether a given configuration is valid. (collides with obstacles)

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        s = np.zeros(2, dtype=np.float32)
        s[0] = node.state[0]
        s[1] = node.state[1]

        if self.geom =='circle':
            for (ox, oy, sizex,sizey) in self.obstacleList:
                obs=[ox+sizex/2.0,oy+sizey/2.0]
                obs_size=[sizex,sizey]
                cf = False
                for j in range(self.dof):
                    if abs(obs[j] - (s[j] + np.sign(obs[j]-s[j])*np.amin([1,abs(obs[j]-s[j])])))>obs_size[j]/2.0:
                        cf=True
                        break
                if cf == False:
                    return False
        elif self.geom=='rectangle':
            for (ox, oy, sizex,sizey) in self.obstacleList:
                obs=[ox+sizex/2.0,oy+sizey/2.0]
                obs_size=[sizex,sizey]

                cf = False
                project=[np.cos(node.angle),np.sin(node.angle)]
                node_size =[3,1.5]
                for j in range(2):
                    if abs(obs[j] - (s[j] + np.sign(obs[j]-s[j])*np.amin([abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]),abs(obs[j]-s[j])])))>obs_size[j]*0.5:
                        cf=True
                        break
                if cf == False:
                    projected_dist= [np.dot(np.array(obs)-np.array(s),np.array([np.cos(node.angle),np.sin(node.angle)])),np.dot(np.array(obs)-np.array(s),np.array([-np.sin(node.angle),np.cos(node.angle)]))]
                    obs_projected = np.array(s) + np.array(projected_dist)

                    project=[np.cos(-node.angle),np.sin(-node.angle)]

                    for j in range(2):
                        if abs(s[j] - (obs_projected[j] + np.sign(s[j]-obs_projected[j])*np.amin([abs(obs_size[0]*0.5*project[j]) + abs(obs_size[1]*0.5*project[1-j]),abs(s[j]-obs_projected[j])])))>node_size[j]*0.5:
                            cf=True
                            break

                if cf == False:
                    return False
        else:
            for (ox, oy, sizex,sizey) in self.obstacleList:
                obs=[ox+sizex/2.0,oy+sizey/2.0]
                obs_size=[sizex,sizey]
                cf = False
                for j in range(self.dof):
                    if abs(obs[j] - s[j])>obs_size[j]/2.0:
                        cf=True
                        break
                if cf == False:
                    return False

        return True  # safe'''

    def get_path_to_goal(self):
        """
        Traverses the tree to chart a path between the start state and the goal state.
        There may be multiple paths already discovered - if so, this returns the shortest one

        Returns: a list of coordinates, representing the path backwards; if a path has been found; None otherwise
        """
        if self.goalfound:
            goalind = None
            mincost = float('inf')
            for idx in self.solutionSet:
                cost = self.nodeList[idx].cost + dist(self.nodeList[idx].state, self.end.state,self.nodeList[idx].angle,self.end.angle)
                if goalind is None or cost < mincost:
                    goalind = idx
                    mincost = cost
            return self.gen_final_course(goalind)
        else:
            return None

    def draw_graph(self, rnd=None,rnd_angle=0):
        """
        Draws the state space, with the tree, obstacles, and shortest path (if found). Useful for visualization.

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for (ox, oy, sizex, sizey) in self.obstacleList:
            rect = mpatches.Rectangle((ox, oy), sizex, sizey, fill=True, color="purple", linewidth=0.1)
            plt.gca().add_patch(rect)


        for node in self.nodeList:
            if node.parent is not None:
                if node.state is not None:
                    plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [
                        node.state[1], self.nodeList[node.parent].state[1]], "-g")

        if self.goalfound:
            path = self.get_path_to_goal()
            x = [p[0][0] for p in path]
            y = [p[0][1] for p in path]
            angle = [p[1] for p in path]
            plt.plot(x, y, '-r',linewidth=3)

            if self.geom=="circle":
                for p in path:
                    circle1 = mpatches.Circle((p[0][0],p[0][1]), 1.0 , fill=True,color="deepskyblue")
                    plt.gca().add_patch(circle1)
            if self.geom=='rectangle':
                for p in path:
                    rec_pos  = np.array([3,1.5])/2
                    rec_pos = np.array(p[0]) - np.matmul(np.array([[np.cos(p[1]),np.sin(-p[1])],[np.sin(p[1]),np.cos(p[1])]]),rec_pos.reshape(2,1)).reshape(2,)
                    rect = mpatches.Rectangle((rec_pos[0],rec_pos[1]), 3, 1.5 , p[1]*180/np.pi,fill=True,color="deepskyblue", linewidth=0.1)
                    plt.gca().add_patch(rect)
                    project=[np.cos(-p[1]),np.sin(-p[1])]
            

        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            if self.geom=="circle":
                circle1 = mpatches.Circle((rnd[0],rnd[1]), 1.0 , fill=True,color="deepskyblue",alpha=0.7)
                plt.gca().add_patch(circle1)
            if self.geom=='rectangle':
                rec_pos  = np.array([3,1.5])/2
                rec_pos = np.array(rnd) - np.matmul(np.array([[np.cos(rnd_angle),np.sin(-rnd_angle)],[np.sin(rnd_angle),np.cos(rnd_angle)]]),rec_pos.reshape(2,1)).reshape(2,)
                rect = mpatches.Rectangle((rec_pos[0],rec_pos[1]), 3, 1.5 , rnd_angle*180/np.pi,fill=True,color="deepskyblue", linewidth=0.1,alpha=0.7)
                plt.gca().add_patch(rect)
                project=[np.cos(-rnd_angle),np.sin(-rnd_angle)]

                node_size =[3,1.5]
                j=1
                plt.plot([rnd[0]-4,rnd[0]+4],[rnd[1]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),rnd[1]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],':r')
                plt.plot([rnd[0]-4,rnd[0]+4],[rnd[1]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),rnd[1]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],':r')
                j=0
                plt.plot([rnd[0]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),rnd[0]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],[rnd[1]-4,rnd[1]+4],':r')
                plt.plot([rnd[0]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),rnd[0]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],[rnd[1]-4,rnd[1]+4],':r')

        plt.plot(self.start.state[0], self.start.state[1], "xr")
        plt.plot(self.end.state[0], self.end.state[1], "xr")
        plt.axis("equal")
        plt.axis([-20, 20, -20, 20])
        plt.grid(True)
        plt.grid(which='minor', alpha=0.2)

        plt.pause(0.01)


class Node():
    """
    RRT Node
    """

    def __init__(self,state,angle=0):
        self.state =state
        self.cost = 0.0
        self.parent = None
        self.children = set()
        self.angle=angle



def main():
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('-g', '--geom', default='point', choices=['point', 'circle', 'rectangle'], \
        help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
    parser.add_argument('--alg', default='rrt', choices=['rrt', 'rrtstar'], \
        help='which path-finding algorithm to use. default: "rrt"')
    parser.add_argument('--iter', default=100, type=int, help='number of iterations to run')
    parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')
    parser.add_argument('--save', action='store_true', help='set to save figures while running script')

    args = parser.parse_args()

    show_animation = not args.blind and not args.fast and not args.save

    print("Starting planning algorithm '%s' with '%s' robot geometry"%(args.alg, args.geom))
    starttime = time.time()


    obstacleList = [
    (-15,0, 15.0, 5.0),
    (15,-10, 5.0, 10.0),
    (-10,8, 5.0, 15.0),
    (3,15, 10.0, 5.0),
    (-10,-10, 10.0, 5.0),
    (5,-5, 5.0, 5.0),
    ]

    start = [-10, -17]
    goal = [10, 10]
    dof=2

    rrt = RRT(start=start, goal=goal, randArea=[-20, 20], obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter)
    path = rrt.planning(animation=show_animation)

    endtime = time.time()

    if path is None:
        print("FAILED to find a path in %.2fsec"%(endtime - starttime))
    else:
        fin_path_cost= RRT.get_path_len(path)
        print("SUCCESS - found path of cost %.5f in %.2fsec"%(fin_path_cost, endtime - starttime))
        if args.blind:  
            with open("results/" + str(rrt.alg) +"-"+ str(rrt.geom) + ".txt",'a+') as file:
                file.seek(0,0)
                num_lines = sum(1 for line in file)
                file.seek(0,2)
                file.write("--> Trial - " + str(round(num_lines/4)+1) + " --------------------------------------------\n")
                file.write("SUCCESS - found path of cost %.5f in %.2fsec"%(fin_path_cost, endtime - starttime)+"\n")
                file.write("Path found - " + str([(round(p[0][0],2),round(p[0][1],2),str(round(p[1]*180/np.pi,2)) + u"\N{DEGREE SIGN}" ) for p in path])+"\n")
                file.write("----------------------------------------------------------\n")
        # Draw final path

    if args.save or not args.blind:
        rrt.draw_graph()
        plt.title("Assignment 1 - " + str(rrt.alg) +" - "+ str(rrt.geom) + "\n Sashank Modali")
    if path is not None:
        plt.title("Assignment 1 - " + str(rrt.alg) +" - "+ str(rrt.geom) + "\n Path cost - " + str(fin_path_cost) + "\n Sashank Modali")
        if args.save:
            plt.savefig("results/plot-"+str(rrt.alg) +"-"+ str(rrt.geom) + ".png")
    if not args.blind:
        plt.show()


if __name__ == '__main__':
    main()
