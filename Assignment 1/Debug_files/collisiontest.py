
import numpy as np
import time
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

def diff(x1,x2):
    return([x1[i]-x2[i] for  i in range(len(x1))])

def __CollisionCheck(s,angle,obstacleList):


    for obstacle in obstacleList:
        obs = [obstacle[0] + obstacle[2]*0.5,obstacle[1]+obstacle[3]*0.5]
        obs_size = [obstacle[2],obstacle[3]]

        cf = False
        project=[np.cos(angle),np.sin(angle)]
        node_size =[3,1.5]
        for j in range(2):
            if abs(obs[j] - (s[j] + np.sign(obs[j]-s[j])*np.amin([abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]),abs(obs[j]-s[j])])))>obs_size[j]*0.5:
                cf=True
                break
        if cf == False:
            projected_dist= [np.dot(np.array(obs)-np.array(s),np.array([np.cos(angle),np.sin(angle)])),np.dot(np.array(obs)-np.array(s),np.array([-np.sin(angle),np.cos(angle)]))]
            obs_projected = np.array(s) + np.array(projected_dist)

            project=[np.cos(-angle),np.sin(-angle)]

            for j in range(2):
                if abs(s[j] - (obs_projected[j] + np.sign(s[j]-obs_projected[j])*np.amin([abs(obs_size[0]*0.5*project[j]) + abs(obs_size[1]*0.5*project[1-j]),abs(s[j]-obs_projected[j])])))>node_size[j]*0.5:
                    cf=True
                    break
        
        if cf == False:
            plt.plot([obs[0]-obs_size[0],obs[0]-obs_size[0],obs[0]-obs_size[0]],[obs[1]-obs_size[1]*0.5,obs[1],obs[1]+obs_size[1]*0.5],'--k')
            plt.plot([obs[0]-obs_size[0]*0.5,obs[0],obs[0]+obs_size[0]*0.5],[obs[1]-obs_size[1],obs[1]-obs_size[1],obs[1]-obs_size[1]],'--k')
            j=1
            plt.plot([obs[0]-obs_size[0]*2,obs[0]-obs_size[0]*2,obs[0]-obs_size[0]*2],[s[1]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),s[1],s[1]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],'-^r')
            j=0
            plt.plot([s[0]-(abs(node_size[j]*0.5*project[j]) + abs(node_size[1-j]*0.5*project[1-j])),s[0],s[0]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],[obs[1]-obs_size[1]*2,obs[1]-obs_size[1]*2,obs[1]-obs_size[1]*2],'->r')
            return False
    return True

def main():

    obstacleList = [
    (-15,0, 15.0, 5.0),
    (15,-10, 5.0, 10.0),
    (-10,8, 5.0, 15.0),
    (3,15, 10.0, 5.0),
    (-10,-10, 10.0, 5.0),
    (5,-5, 5.0, 5.0),
    ]


    start = time.time()
    duration = 0
    step=0
    while duration <120 :
        if duration>step:
            print(str(duration) + " seconds have passed")
            step+=10
        s=[]

        s.append(np.random.uniform(-15,20))
        s.append(np.random.uniform(-15,20))

        angle = np.random.uniform(0,2*np.pi)

        flag=__CollisionCheck(s,angle,obstacleList)

        plt.plot(s[0],s[1],'-xr')

        rec_pos  = np.array([3,1.5])/2

        rec_pos = np.array(s) - np.matmul(np.array([[np.cos(angle),np.sin(-angle)],[np.sin(angle),np.cos(angle)]]),rec_pos.reshape(2,1)).reshape(2,)

        for (ox, oy, sizex, sizey) in obstacleList:
            rect = mpatches.Rectangle((ox, oy), sizex, sizey, fill=True, color="purple", linewidth=0.1)
            plt.gca().add_patch(rect)
        rect = mpatches.Rectangle((rec_pos[0], rec_pos[1]), 3, 1.5, angle*180/np.pi, fill=True, color="red", linewidth=0.1)
        project=[np.cos(angle),np.sin(angle)]
        node_size =[3,1.5]

        j=1
        plt.plot([-20,20],[s[1]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),s[1]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],':r')
        plt.plot([-20,20],[s[1]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),s[1]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],':r')
        j=0
        plt.plot([s[0]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),s[0]+(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],[-20,20],':r')
        plt.plot([s[0]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j])),s[0]-(abs(node_size[0]*0.5*project[j]) + abs(node_size[1]*0.5*project[1-j]))],[-20,20],':r')

        
        plt.gca().add_patch(rect)
        plt.title("Didn't Collide = " + str(flag) + " , angle = " + str(np.round(angle*180/np.pi,2)))
        plt.axis("equal")
        plt.axis([-20, 20, -20, 20])
        plt.grid(True)
        plt.pause(10)
        plt.clf()
        duration = time.time()-start

if __name__=="__main__":
    plt.show()
    main()