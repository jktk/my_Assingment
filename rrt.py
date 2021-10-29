
import numpy as np
import cv2
import matplotlib.pyplot as plt
from makepath import makefloor
from combine import frames_to_video
import os 
for file in os.listdir("./holonomic/"):
    if file.endswith(".png"):
        os.system("rm ./holonomic/*.png")
        break
floorplan , erosion = makefloor((600 , 600) , 30)

class rrttree:
    def __init__(self, x, y, nodeno, parent):
        self.x = x
        self.y = y
        self.nodeno = nodeno
        self.parent = parent


N = 600
step_size = 18
threshold = 20
x_max = 600
y_max = 600
bias = 0.05

x_goal = 400
y_goal = 500
rrt = rrttree(50, 50, 1, 0)
floorplan = cv2.circle(floorplan, (50, 50), 8, (0, 255, 0), -1)
floorplan = cv2.circle(floorplan, (x_goal, y_goal), 8, (0, 0, 255), -1)

rrtT = {}
rrtT[rrt.nodeno] = rrt

iter = 2

def getdistance(rrtT , x_rand  , y_rand ):
    # return the min distance and the index for it.
    # return distance  , parentind
    min_dist=(y_max*x_max)+1
    index=0
    for i in range(len(rrtT)):
        # tempnode=i
        xt,yt= rrtT[i+1].x, rrtT[i+1].y
        temp_dist=np.sqrt((x_rand-xt)**2 + (y_rand-yt)**2)
        if temp_dist<=min_dist:
            min_dist=temp_dist
            index=rrtT[i+1].nodeno
    return min_dist, index


def goalreached(rrtnode):
    # check if the goal is reached from the generated node, then return true
    distance = np.sqrt((rrtnode.x-x_goal)**2 + (rrtnode.y-y_goal)**2)
    if distance <= threshold:
        rrtgoal = rrttree(x_goal, y_goal, iter+1, iter)
        rrtT[iter+1] = rrtgoal
        return True
    else:
        return False


def drawline(rrt, parentind , f=0):
    if f == 0 :
        cv2.line(floorplan, (rrt.x, rrt.y),
                (rrtT[parentind].x, rrtT[parentind].y), (125, 125, 0), 1)
        
        cv2.circle(floorplan , (rrt.x, rrt.y) , 3 , (125, 125 , 0 ) ,-1)
    else:
        cv2.line(floorplan, (rrt.x, rrt.y),
                (rrtT[parentind].x, rrtT[parentind].y), (1, 0, 255), 1)
        cv2.circle(floorplan , (rrt.x, rrt.y) , 3 , (0, 0 , 255 ) ,-1)


def checkobstical(x_new, y_new, erosion, parentind):
    x_old = rrtT[parentind].x
    y_old = rrtT[parentind].y
    if x_old  > x_new :
        for i in range(x_new , x_old+1 ):
            y  = np.int(np.round( (y_old - y_new ) * (i - x_new)/(x_old - x_new ) + y_new ))
            if np.mean(erosion[y , i]) == 0 :
                return True
    elif x_old  < x_new:
        for i in range(x_old , x_new+1 ):
            y  = np.int(np.round( (y_old - y_new ) * (i - x_new)/(x_old - x_new ) + y_new ))
            if np.mean(erosion[y , i]) == 0 :
                return True
    else :
        # print("fine" , parentind)
        return True 

    return False


while iter < N:
    if np.random.uniform(0, 1, 1)[0] < bias:
        x_rand = x_goal
        y_rand = y_goal
    else:
        x_rand = np.round(np.random.uniform(0, x_max))
        y_rand = np.round(np.random.uniform(0, y_max))

    distance, parentind = getdistance(rrtT , x_rand , y_rand )

    if distance >= step_size:
        x_new = rrtT[parentind].x + \
            ((x_rand - rrtT[parentind].x)*step_size)/distance
        y_new = rrtT[parentind].y + \
            ((y_rand - rrtT[parentind].y)*step_size)/distance
    else:
        x_new = x_rand
        y_new = y_rand
    x_new = np.int(np.round(x_new))
    y_new = np.int(np.round(y_new))

    if checkobstical(x_new, y_new, erosion, parentind):
        continue

    rrt = rrttree(x_new, y_new, iter, parentind)

    rrtT[rrt.nodeno] = rrt
    drawline(rrt, parentind)

    if goalreached(rrt):
        break
    iter += 1
    cv2.imwrite('./holonomic/'  + str(iter) + '.png' , floorplan)

while(iter < N and rrtT[iter].parent != 0 ):
    drawline(rrtT[iter] , rrtT[iter].parent , 1)
    iter = rrtT[iter].parent


cv2.imwrite('./holonomic/'  + str(N+3) + '.png' , floorplan)
frames_to_video('./holonomic/*.png' , './holonomic/holonomic.mp4' , 25)
# plt.imshow(floorplan)
# plt.show()
