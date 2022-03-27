
from builtins import object
from matrix import matrix
import random
import numpy as np
import math
import pdb

state_num = 6
init_covariance = 1000
dt = 0.1
# external motion
u = matrix([[0.], [0.], [0.], [0.], [0.], [0.]])
# next state function
F =  matrix([[1., dt, 0.5*( dt**2 ), 0., 0., 0.], [0., 1., dt, 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 1., dt, 0.5*( dt**2 )], [0., 0., 0., 0., 1., dt], [0., 0., 0., 0., 0., 1.]])
single_integrator_F =  matrix([[1., dt, 0, 0., 0., 0.], [0., 1., dt, 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 1., dt, 0], [0., 0., 0., 0., 1., dt], [0., 0., 0., 0., 0., 1.]])
# measurement function: reflect the fact that we observe x and y 
H =  matrix([[1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]])
# measurement uncertainty: use 6x6 matrix with 0 as main diagonal
R =  matrix([[0.01, 0.], [0., 0.01]])  #matrix([[0.1, 0.], [0., 0.1]]) 
# 6d identity matrix
I =  matrix(np.eye(state_num)) 

class KalmanFilter(object):

    class ObstacleMeasure(object):        
        def __init__(self):
            # x, x', x'', y, y', y''
            self.estimate = matrix([np.array([0] * state_num)]).transpose()
            # init covariance between x, x', x'', y, y', y''
            self.covariance = matrix(np.eye(state_num) * init_covariance)

    def __init__(self):
        self.obstacle_locations = []
        self.obstacle_measure = {}
        
    def observe_obstacles(self, obstacle_locations):
        """ self - pointer to the current object.
           asteroid_locations - a list of asteroid observations. Each 
           observation is a tuple (i,x,y) where i is the unique ID for
           an asteroid, and x,y are the x,y locations (with noise) of the
           current observation of that asteroid at this timestep.
           Only asteroids that are currently 'in-bounds' will appear
           in this list, so be sure to use the asteroid ID, and not
           the position/index within the list to identify specific
           asteroids. (The list may change in size as asteroids move
           out-of-bounds or new asteroids appear in-bounds.)

           Return Values:
                    None
        """
        measure_locations = []
        for location in obstacle_locations:
            ID, x, y = location
            observation = matrix([[x], [y]])
            if (ID not in self.obstacle_measure):
                self.obstacle_measure[ID] = self.ObstacleMeasure()
            estimate = self.obstacle_measure[ID].estimate
            covariance = self.obstacle_measure[ID].covariance
            # measurement update
            y = observation - (H * estimate)            
            S = H * (covariance) * H.transpose() + R
            K = covariance * H.transpose() * S.inverse()
            #pdb.set_trace()
            estimate = estimate + (K * y)
            #print(f"covariance {covariance}")
            #pdb.set_trace()
            covariance = (I - (K * H)) * covariance
            
            # add asteroids 
            measure_locations.append((ID, estimate[0][0], estimate[3][0]))
            self.obstacle_measure[ID].estimate = estimate                
            self.obstacle_measure[ID].covariance = covariance
        
        self.obstacle_locations = measure_locations

    def distance(self, robot_x, robot_y, obs_x, obs_y):
        """        
        return: float value - means the distance between craft and asteroid
        """
        diff_x = robot_x - obs_x
        diff_y = robot_y - obs_y
        return math.sqrt(diff_x ** 2 + diff_y ** 2)
    
    def estimate_obstacle_locs(self):
        """ Should return an iterable (list or tuple for example) that
            contains data in the format (i,x,y), consisting of estimates
            for all in-bound asteroids. """
        estimate_locations = []
        for location in self.obstacle_locations:
            ID, x, y = location
            if (ID not in self.obstacle_measure):
                self.obstacle_measure[ID] = self.ObstacleMeasure()
            estimate = self.obstacle_measure[ID].estimate
            covariance = self.obstacle_measure[ID].covariance
            # prediction
            estimate = (F * estimate)
            covariance = F * covariance * F.transpose()
            
            estimate_locations.append((ID, estimate[0][0], estimate[3][0]))
            self.obstacle_measure[ID].estimate = estimate
            self.obstacle_measure[ID].covariance = covariance     

        return estimate_locations

    def single_integrator_estimate_obstacle_locs(self):
        """ Should return an iterable (list or tuple for example) that
            contains data in the format (i,x,y), consisting of estimates
            for all in-bound asteroids. """
        estimate_locations = []
        for location in self.obstacle_locations:
            ID, x, y = location
            if (ID not in self.obstacle_measure):
                self.obstacle_measure[ID] = self.ObstacleMeasure()
            estimate = self.obstacle_measure[ID].estimate
            covariance = self.obstacle_measure[ID].covariance
            # prediction
            estimate = (single_integrator_F * estimate)
            covariance = F * covariance * F.transpose()
            
            estimate_locations.append((ID, estimate[0][0], estimate[3][0]))
            self.obstacle_measure[ID].estimate = estimate
            self.obstacle_measure[ID].covariance = covariance     

        return estimate_locations
    '''
    def estimate_craft_path(self, x, y, vel, angle):
        """
        x - craft start x position
        y - craft start y position
        vel - craft start vel position
        angle - craft start angle position
        
        return - craft path
        """
        craft_path = []
        for i in range(FORECAST_STEP):
            x = x + (vel * math.cos( angle ))
            y = y + (vel * math.sin( angle ))
            craft_path.append((x, y))
        return craft_path
    
    def score(self, craft_path, asteroid_paths):
        """
        craft_path - path craft will go in next FORECAST_STEP
        asteroid_paths - paths that dangerous asteroids will go in next FORECAST_STEP
        
        return - float value score represent how safe this craft path is, 
            larger the return value is, safer this path is
        """       
        score = 0
        for asteroid_path in asteroid_paths:
            for i in range(len(asteroid_path)):
                asteroid_pos = asteroid_path[i]
                craft_pos = craft_path[i]
                distance = self.distance(craft_pos[0], craft_pos[1], asteroid_pos[0], asteroid_pos[1])
                if (distance < self.min_dist * 2):
                    distance = 2 * (distance * -1)
                score += distance
        return score
    
    
    def next_move(self, craft_state):
        """ self - a pointer to the current object.
            craft_state - implemented as CraftState in craft.py.

            return values: 
              angle change: the craft may turn left(1), right(-1), 
                            or go straight (0). 
                            Turns adjust the craft's heading by 
                             angle_increment.
              speed change: the craft may accelerate (1), decelerate (-1), or 
                            continue at its current velocity (0). Speed 
                            changes adjust the craft's velocity by 
                            speed_increment, maxing out at max_speed.
         """
        self.craft = craft_state
        self.possible_speeds = [self.craft.v]
        low_speed = self.craft.v - self.craft.speed_increment
        if (low_speed > 0):
            self.possible_speeds.append(low_speed)
        high_speed = self.craft.v + self.craft.speed_increment              
        self.possible_speeds.append(min(self.craft.max_speed, high_speed))
        self.possible_angles = [self.craft.h]
        left_angle = self.craft.h - self.craft.angle_increment
        right_angle = self.craft.h + self.craft.angle_increment
        if (0 < left_angle < math.pi):
            self.possible_angles.append(left_angle)
        if (0 < right_angle < math.pi):
            self.possible_angles.append(right_angle)
          
        # estimate asteroid position
        self.estimate_asteroid_locs()                                
        danger_angles = []
        safe_asteroids = []
        
        if (len(self.closing_asteroid) > 0):
            # estimate asteroid path
            asteroid_paths = []                    
            for ID, asteroid_measure in self.closing_asteroid.items():
                if (self.distance(self.craft.x, self.craft.y, asteroid_measure.estimate[0][0], asteroid_measure.estimate[3][0]) > self.safty_distance):
                    safe_asteroids.append(ID)
                    continue
                path = [(asteroid_measure.estimate[0][0], asteroid_measure.estimate[3][0])]
                next_estimate = asteroid_measure.estimate
                for i in range(1, FORECAST_STEP):
                    next_estimate = F * next_estimate
                    path.append((next_estimate[0][0], next_estimate[3][0]))
                asteroid_paths.append(path)
                
            # estimate craft path and score each path
            score_map = {}
            safe = False
            for angle in self.possible_angles:
                for vel in self.possible_speeds:
                    craft_path = self.estimate_craft_path(self.craft.x, self.craft.y, vel, angle)
                    score = self.score(craft_path, asteroid_paths)
                    score += vel / 5
                    score_map[(angle, vel)] = score - (abs(angle - DIRECT_ANGLE) / 5)          
            safest_angle, safest_vel = max(score_map, key = score_map.get)
            # get increment            
            angle_increment = safest_angle - craft_state.h
            vel_increment = safest_vel - craft_state.v
            for ID in safe_asteroids:
                self.closing_asteroid.pop(ID)
             
        else:    
            angle_increment = DIRECT_ANGLE - craft_state.h
            vel_increment = MAX_SPEED - craft_state.v
            
        if (angle_increment > 0):
            angle_increment = 1
        elif (angle_increment < 0):
            angle_increment = -1  
        if (vel_increment > 0):
            vel_increment = 1
        elif (vel_increment < 0):
            vel_increment = -1 
            
        return angle_increment, vel_increment
        ''' 
