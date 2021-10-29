import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import _reshape_dispatcher
from makepath import makefloor
from combine import frames_to_video

sys.stdout = open("output.txt", "w")
os.system("rm non_holonomic/*")
os.system("rm non_holonomic_wheel/*")

class rrttreenon:
    def __init__(self, x, y, nodeno, parent, theta, parent_s, p_v):
        self.x = x
        self.y = y
        self.theta = theta
        self.nodeno = nodeno
        self.parent = parent
        self.parent_s = parent_s
        self.p_v = p_v


N = 1000
step_size = 15
threshold = 20
x_max = 600
y_max = 600
bias = 0.01
x_goal = 400
y_goal = 500
orientation_threshold = 30*np.pi/180
dt = 0.1
lrobot = 20
wrobot = 10
diagofrobot = np.int(np.ceil(np.sqrt(lrobot**2 + wrobot**2)))
maxsteeing = 0.7
max_v = 40
theta_goal = 90*np.pi/180

rrt = rrttreenon(50, 50, 1, 0, 0, 90*np.pi/180, 0)

floorplan, erosion = makefloor((x_max, y_max), diagofrobot)
wheelplan = np.copy(floorplan)

floorplan = cv2.circle(floorplan, (50, 50), 8, (0, 255, 0), -1)
floorplan = cv2.circle(floorplan, (x_goal, y_goal), 20, (0, 0, 255), -1)

wheelplan = cv2.circle(wheelplan, (50, 50), 8, (0, 255, 0), -1)
wheelplan = cv2.circle(wheelplan, (x_goal, y_goal), 20, (0, 0, 255), -1)
rrtT = {}
rrtT[rrt.nodeno] = rrt


def goalreached(rrtnode):
    '''
    this is to check if the goal is reached. If yes then add the final edge between
    the node and the final destination.
    return true or false
    '''
    distance = np.sqrt((rrtnode.x-x_goal)**2 + (rrtnode.y-y_goal)**2)
    theta_diff = min(abs(rrtnode.theta - theta_goal),
                     abs(rrtnode.theta - theta_goal - np.pi))
    if distance <= threshold and theta_diff <= orientation_threshold:
    # if distance <= threshold:
        rrtgoal = rrttreenon(x_goal, y_goal, iter+1, iter,
                             theta_goal, maxsteeing, max_v)
        rrtT[iter+1] = rrtgoal
        return True
    else:
        return False
    pass


def getdistance(rrtT, x_rand, y_rand, theta_rand):
    '''
    get the distance along the curve from the node which is closest to 
    the randomly selected node .
    return distance , new_parentind 

    '''
    min_dist = (y_max*x_max)+1 + (360)**2
    index = 0
    for i in range(len(rrtT)):
        xt, yt = rrtT[i+1].x, rrtT[i+1].y
        temp_dist = np.sqrt((x_rand-xt)**2 + (y_rand-yt)**2 +
                            ((180/np.pi)**2)*min((theta_rand - rrtT[i+1].theta)**2, (theta_rand - rrtT[i+1].theta - np.pi)**2,
                                                 (theta_rand - rrtT[i+1].theta + np.pi)**2))
        if temp_dist <= min_dist:
            min_dist = temp_dist
            index = rrtT[i+1].nodeno
    # print(index)
    return min_dist, index


def drawcurve(path, lin_v, steering_angle, f=0):
    """
    function to draw the curve 

    """
    
    if f == 0:
        
        for i in range(step_size):
            cv2.line(floorplan, (np.int(np.round(path[0])), np.int(np.round(path[1]))),
                     (np.int(np.round(path[0] + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1] + lin_v * np.sin(path[2])*dt))), (125, 255,0), 1)
            cv2.line(wheelplan, (np.int(np.round(path[0])), np.int(np.round(path[1]))),
                     (np.int(np.round(path[0] + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1] + lin_v * np.sin(path[2])*dt))), (125, 255,0), 1)
            cv2.line(wheelplan, (np.int(np.round(path[0]+ 5*np.cos(path[2] + np.pi/2))), np.int(np.round(path[1] +5* np.sin(path[2] + np.pi/2)))),
                     (np.int(np.round(path[0] + 5*np.cos(path[2] + np.pi/2) + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1] + 5*np.sin(path[2] + np.pi/2)+ lin_v * np.sin(path[2])*dt))), (125, 255,0), 1)
            cv2.line(wheelplan, (np.int(np.round(path[0]+5* np.cos(path[2] - 5*np.pi/2))), np.int(np.round(path[1] + 5*np.sin(path[2] - np.pi/2) ))),
                     (np.int(np.round(path[0]+5* np.cos(path[2] - np.pi/2) + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1]+ 5*np.sin(path[2] - np.pi/2) + lin_v * np.sin(path[2])*dt))), (125, 255,0), 1)
            
            path[0] += lin_v * np.cos(path[2])*dt
            path[1] += lin_v * np.sin(path[2])*dt
            path[2] += (lin_v / lrobot) * np.tan(steering_angle) * dt
            if np.mean(erosion[np.int(np.ceil(path[1])), np.int(np.ceil(path[0]))]) == 0 or np.mean(erosion[np.int(np.floor(path[1])), np.int(np.floor(path[0]))]) == 0:
                return -1
        cv2.circle(floorplan,  (np.int(np.round(path[0])), np.int(
            np.round(path[1]))), 3, ( 0 , 255,125), -1)
    else:
        for i in range(step_size):
            cv2.line(floorplan, (np.int(np.round(path[0])), np.int(np.round(path[1]))),
                     (np.int(np.round(path[0] + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1] + lin_v * np.sin(path[2])*dt))), (255, 0,0), 1)
            cv2.line(wheelplan, (np.int(np.round(path[0]+ 5*np.cos(path[2] + np.pi/2))), np.int(np.round(path[1] +5* np.sin(path[2] + np.pi/2)))),
                     (np.int(np.round(path[0] + 5*np.cos(path[2] + np.pi/2) + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1] + 5*np.sin(path[2] + np.pi/2)+ lin_v * np.sin(path[2])*dt))), (255, 0,0), 1)
            cv2.line(wheelplan, (np.int(np.round(path[0]+5* np.cos(path[2] - 5*np.pi/2))), np.int(np.round(path[1] + 5*np.sin(path[2] - np.pi/2) ))),
                     (np.int(np.round(path[0]+5* np.cos(path[2] - np.pi/2) + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1]+ 5*np.sin(path[2] - np.pi/2) + lin_v * np.sin(path[2])*dt))), (255, 0,0), 1)
            cv2.line(wheelplan, (np.int(np.round(path[0])), np.int(np.round(path[1]))),
                     (np.int(np.round(path[0] + lin_v * np.cos(path[2])*dt)), np.int(np.round(path[1] + lin_v * np.sin(path[2])*dt))), (255, 0,0), 1)
            
            path[0] += lin_v * np.cos(path[2])*dt
            path[1] += lin_v * np.sin(path[2])*dt
            path[2] += (lin_v / lrobot) * np.tan(steering_angle) * dt

        cv2.circle(floorplan, (np.int(np.round(path[0])), np.int(
            np.round(path[1]))), 3, (255, 0, 0), -1)


def getnewnode(rrtT, parentind, x_rand, y_rand, theta_rand):
    path = [rrtT[parentind].x, rrtT[parentind].y, rrtT[parentind].theta]
    steering_angle = -maxsteeing
    lin_v = -max_v
    distance_new = (y_max*x_max)+1 + (360)**2 + 3
    final_path = []
    is_valid = 0
    while lin_v <= max_v:

        steering_angle = -maxsteeing
        while steering_angle <= maxsteeing:
            path = [rrtT[parentind].x, rrtT[parentind].y, rrtT[parentind].theta]

            for _ in range(step_size):
                is_valid = 0
                path[0] += lin_v * np.cos(path[2])*dt
                path[1] += lin_v * np.sin(path[2])*dt
                path[2] += (lin_v / lrobot) * np.tan(steering_angle) * dt
                if np.int(np.floor(path[0])) < 0 or np.int(np.ceil(path[0])) >= y_max:
                    is_valid = 1
                    break

                if np.int(np.floor(path[1])) < 0 or np.int(np.ceil(path[1])) >= x_max:
                    is_valid = 1
                    break

                if np.mean(erosion[np.int(np.ceil(path[1])), np.int(np.ceil(path[0]))]) == 0 or np.mean(erosion[np.int(np.floor(path[1])), np.int(np.floor(path[0]))]) == 0:
                    is_valid = 1
                    break

            if is_valid == 1:
                steering_angle += 0.05
                continue
            temp_dist = np.sqrt((x_rand - path[0])**2 + (y_rand-path[1])**2 + ((180/np.pi)**2)*min(
                (theta_rand - path[2])**2, (theta_rand - path[2] - np.pi)**2, (theta_rand - path[2] + np.pi)**2))

            if distance_new > temp_dist:
                distance_new = temp_dist
                final_path = [lin_v, steering_angle, path]
            steering_angle += 0.05

        if (lin_v + 5) > -30 and (lin_v + 5) < 30:
            lin_v = 30
        else:
            lin_v = lin_v + 5

    path = [rrtT[parentind].x, rrtT[parentind].y, rrtT[parentind].theta]
    if len(final_path) < 1:
        return None

    test = drawcurve(path, final_path[0], final_path[1])
    if test == -1:
        return []
    else :
        return final_path


iter = 2

while iter < N:
    if np.random.uniform(0, 1, 1)[0] < bias:
        x_rand = x_goal
        y_rand = y_goal
        theta_rand = theta_goal
    else:
        x_rand = np.round(np.random.uniform(0, x_max))
        y_rand = np.round(np.random.uniform(0, y_max))
        theta_rand = np.round(np.random.uniform(0.001, 2*np.pi))

    distance, parentind = getdistance(rrtT, x_rand, y_rand, theta_rand)
    if distance < 0.1:
        continue

    final_path = getnewnode(rrtT, parentind,  x_rand, y_rand, theta_rand)

    if final_path == None:
        continue

    rrt = rrttreenon(final_path[2][0], final_path[2][1], iter,
                        parentind, final_path[2][2], final_path[1], final_path[0])

    rrtT[rrt.nodeno] = rrt

    if goalreached(rrt):
        break
    iter += 1
    cv2.imwrite('./non_holonomic/' + str(iter) + '.png', floorplan)
    cv2.imwrite('./non_holonomic_wheel/' + str(iter) + '.png', wheelplan)
while(iter < N and rrtT[iter].parent != 0):
    parentind = rrtT[iter].parent
    path = [rrtT[parentind].x, rrtT[parentind].y, rrtT[parentind].theta]
    drawcurve(path, rrtT[iter].p_v, rrtT[iter].parent_s, f=1)
    iter = rrtT[iter].parent

plt.imshow(floorplan)
plt.figure()
plt.imshow(wheelplan)
plt.show()

cv2.imwrite('./non_holonomic/'  + str(N+3) + '.png' , floorplan)
cv2.imwrite('./non_holonomic_wheel/' + str(N+ 3) + '.png', wheelplan)

frames_to_video('./non_holonomic/*.png' , './non_holonomic/non_holonomic.mp4' , 20)
frames_to_video('./non_holonomic_wheel/*.png' , './non_holonomic_wheel/non_holonomicwheel.mp4' , 20)
