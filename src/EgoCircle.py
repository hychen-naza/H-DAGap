import math
import numpy as np
from operator import attrgetter
import pdb
import copy

K_EPSILON = 5.9915

class LaserData:
    def __init__(self, depth: float, theta: float, rel_x: float, rel_y: float, v_x: float, v_y: float, id = -1):
        self.x = rel_x
        self.y = rel_y
        self.vx = v_x
        self.vy = v_y
        self.depth = depth
        self.theta = theta
        self.id = id
    
    def calcu_uncertainty_bound(self, sensor_data):
        covariance = sensor_data['obstacle_sensor_est']['obs'+str(self.id)]['covariance']
        val, _ = np.linalg.eig(np.array(covariance.value)[:2,:2])
        uncertainty_bound = 0
        for k in range(len(val)):
            uncertainty_bound += np.sqrt(K_EPSILON * val[k])
        return min(0.09, uncertainty_bound*0.1)

class Gap:
    def __init__(self, x, y, left_laserdata = None, right_laserdata = None):
        # the coordinate of mid_point
        self.x = x 
        self.y = y
        # the left and right obstacles
        self.left_obs = left_laserdata
        self.right_obs = right_laserdata
        self.id = str(left_laserdata.id) + '_' + str(right_laserdata.id)
    
    def set_score(self, score):
        self.score = score

    def update(self, dt):
        self.left_obs.x = self.left_obs.x + dt * self.left_obs.vx
        self.left_obs.y = self.left_obs.y + dt * self.left_obs.vy
        self.right_obs.x = self.right_obs.x + dt * self.right_obs.vx
        self.right_obs.y = self.right_obs.y + dt * self.right_obs.vy

class EgoCircle:
    def __init__(self, goal: list, dist: float, radius: float, safety_dist: float):
        self.depths = {}
        self.inflated_depths = {}       
        self.goal = np.array(goal)
        self.dist = dist # intermedia goal dist
        self.radius = radius # laser sensing range
        self.safety_dist = safety_dist
        self.angle_interval = 90

    def calcu_angle(self, rel_x, rel_y):
        angle = math.atan2(rel_x, rel_y)
        angle = int(angle * 180 / math.pi)
        if (angle < 0):
            angle += 360
        return angle

    def parse_sensor_data(self, sensor_data: dict):
        robot_pos_state = sensor_data['cartesian_sensor_est']['pos'].flatten()
        robot_vel_state = sensor_data['cartesian_sensor_est']['vel'].flatten()
        obstacle_states = sensor_data['obstacle_sensor_est']
        # transform to laser form data
        for _, obs_state in obstacle_states.items():
            obs_pos_state = obs_state['pos'].flatten()
            obs_vel_state = obs_state['vel'].flatten()
            rel_x = obs_pos_state[0] - robot_pos_state[0]
            rel_y = obs_pos_state[1] - robot_pos_state[1]
            depth = np.linalg.norm([rel_x, rel_y])
            angle = self.calcu_angle(rel_x, rel_y)
            if (angle not in self.depths):
                self.depths[angle] = []
            self.depths[angle].append(LaserData(depth, angle, rel_x, rel_y, obs_vel_state[0], obs_vel_state[1], obs_state['id']))

        # pick the closest one in every egocircle bucket
        for angle, laser_datas in self.depths.items():           
            self.inflated_depths[angle] = min(laser_datas, key=attrgetter('depth'))

    def build_gaps(self, sensor_data: dict):
        robot_state = sensor_data['cartesian_sensor_est']['pos'].flatten()        
        angles = list(self.inflated_depths.keys())
        angles.sort()        
        gaps = []
        fake_obs_id = -1
        # no obstacle, move towards the goal
        if (len(angles) == 0):
            depth = self.radius
            angle = math.ceil(2*self.safety_dist*360 / (2*math.pi*depth))+5
            theta = math.ceil(angle / 2)    
            angles = [-theta, theta]
            for fake_theta in angles:
                rel_x = depth * math.sin(fake_theta * 2 * math.pi / 360)
                rel_y = depth * math.cos(fake_theta * 2 * math.pi / 360)
                self.inflated_depths[fake_theta] = LaserData(depth, fake_theta, rel_x, rel_y,0,0,fake_obs_id)
                fake_obs_id -= 1
            gaps = self.build_gap_from_angles(angles, robot_state)
        # one obstacle
        elif (len(angles) == 1):
            laser_data = self.inflated_depths[angles[0]]            
            #angle = math.ceil(2*self.safety_dist*360 / (2*math.pi*laser_data.depth))
            num_gap = int(360 / self.angle_interval)
            angles = [laser_data.theta]
            for i in range(1, num_gap+1):
                fake_theta = (laser_data.theta + i * self.angle_interval) % 360
                angles.append(fake_theta)
                rel_x = self.radius * math.sin(fake_theta * 2 * math.pi / 360)
                rel_y = self.radius * math.cos(fake_theta * 2 * math.pi / 360)
                self.inflated_depths[fake_theta] = LaserData(self.radius, fake_theta, rel_x, rel_y, 0, 0,fake_obs_id)
                fake_obs_id -= 1
            gaps = self.build_gap_from_angles(angles, robot_state)
        # multiple obstacles
        else:
            angle_start = copy.deepcopy(angles)
            angle_end = copy.deepcopy(angles[1:])
            angle_end.append(angles[0])
            for angle1, angle2 in zip(angle_start, angle_end):  
                # split the big empty chunk              
                laser_data1 = self.inflated_depths[angle1]
                laser_data2 = self.inflated_depths[angle2]
                #min_depth = min(laser_data1.depth, laser_data2.depth) # for simplicity, use the min depth
                #angle = math.ceil(2*self.safety_dist*360 / (2*math.pi*min_depth))
                diff_angle = angle2 - angle1
                if (diff_angle < 0):
                    diff_angle += 360
                num_gap = int(diff_angle / self.angle_interval)
                if (num_gap >= 2):
                    for i in range(1, num_gap):
                        fake_theta = (angle1 + i * self.angle_interval) % 360
                        angles.append(fake_theta)
                        #depth = laser_data1.depth + ((i * self.angle_interval) / (diff_angle) * (laser_data2.depth - laser_data1.depth))
                        rel_x = self.radius * math.sin(fake_theta * 2 * math.pi / 360)
                        rel_y = self.radius * math.cos(fake_theta * 2 * math.pi / 360)
                        self.inflated_depths[fake_theta] = LaserData(self.radius, fake_theta, rel_x, rel_y, 0, 0,fake_obs_id)
                        fake_obs_id -= 1
            angles.sort()  
            gaps = self.build_gap_from_angles(angles, robot_state, sensor_data)
        self.inflated_depths = {}
        self.depths = {}
        return gaps

    def build_gap_from_angles(self, angles, robot_state, sensor_data):
        angle_start = angles
        angle_end = angles[1:]
        angle_end.append(angles[0])
        gaps = []
        best_gap = None
        max_arc_dist = float("-inf")
        for angle1, angle2 in zip(angle_start, angle_end):                
            laser_data1 = self.inflated_depths[angle1]
            laser_data2 = self.inflated_depths[angle2]
            bound1 = laser_data1.calcu_uncertainty_bound(sensor_data)
            bound2 = laser_data2.calcu_uncertainty_bound(sensor_data)
            diff_angle = angle2 - angle1
            if (diff_angle < 0):
                diff_angle += 360
            mid_angle = angle1 + diff_angle / 2                
            avg_depth = (laser_data1.depth + laser_data2.depth) / 2
            arc_dist = 2*math.pi*avg_depth*(diff_angle/360)
            rel_x = (laser_data1.x + laser_data2.x) / 2 #avg_depth * math.sin(mid_angle * 2 * math.pi / 360)
            rel_y = (laser_data1.y + laser_data2.y) / 2 #avg_depth * math.cos(mid_angle * 2 * math.pi / 360)    
            #pdb.set_trace()
            #if (diff_angle > 60 and np.linalg.norm([laser_data1.x - laser_data2.x, laser_data1.y - laser_data2.y]) > 2*self.safety_dist):
            # Large interval
            if (arc_dist > 2*self.safety_dist+bound1+bound2):                                
                gaps.append(Gap(rel_x, rel_y, laser_data1, laser_data2))
            if (arc_dist > max_arc_dist):
                max_arc_dist = arc_dist
                best_gap = Gap(rel_x, rel_y, laser_data1, laser_data2)
        if (len(gaps) == 0):
            gaps.append(best_gap)
        return gaps

    def select_gap(self, sensor_data, gaps):
        robot_state = sensor_data['cartesian_sensor_est']['pos'].flatten()
        goal_rel_pos = self.goal - robot_state[:2]
        goal_rel_pos = self.dist * goal_rel_pos / (np.linalg.norm(goal_rel_pos, 2)) 
        goal_x = goal_rel_pos[0]
        goal_y = goal_rel_pos[1]
        best_gap = None
        min_dist = float("inf")
        for gap in gaps:
            dist = np.linalg.norm([goal_x - gap.x, goal_y - gap.y])
            if (dist < min_dist):
                min_dist = dist
                best_gap = gap
        intermedia_goal = np.vstack([best_gap.x, best_gap.y, 0, 0])
        return intermedia_goal, best_gap

    def select_dynamic_gap(self, sensor_data, gaps):
        robot_state = sensor_data['cartesian_sensor_est']['pos'].flatten()
        goal_rel_pos = self.goal - robot_state[:2]
        goal_rel_pos = self.dist * goal_rel_pos / (np.linalg.norm(goal_rel_pos, 2)) 
        goal_x = goal_rel_pos[0]
        goal_y = goal_rel_pos[1]
        best_gap = None
        max_score = float("-inf")
        discount_rate = 0.9
        dt = 0.1
        
        for gap in gaps:
            left_obs = gap.left_obs
            right_obs = gap.right_obs
            score = 0
            if (left_obs is not None and right_obs is not None):
                for i in range(10):                
                    right_rel_x = right_obs.x + i * dt * right_obs.vx
                    right_rel_y = right_obs.y + i * dt * right_obs.vy
                    right_depth = np.linalg.norm([right_rel_x, right_rel_y])
                    right_angle = self.calcu_angle(right_rel_x, right_rel_y)

                    left_rel_x = left_obs.x + i * dt * left_obs.vx
                    left_rel_y = left_obs.y + i * dt * left_obs.vy
                    left_depth = np.linalg.norm([left_rel_x, left_rel_y])
                    left_angle = self.calcu_angle(left_rel_x, left_rel_y)

                    # measure safety score
                    diff_angle = right_angle - left_angle
                    if (diff_angle < 0):
                        diff_angle += 360
                    mid_angle = left_angle + diff_angle / 2                
                    avg_depth = (right_depth + left_depth) / 2
                    arc_dist = 2*math.pi*avg_depth*(diff_angle/360)                
                    score += arc_dist * (discount_rate ** i)
                    # measure efficiency score
                    goal_y = avg_depth * math.cos(mid_angle * 2 * math.pi / 360)  
                    score += 3 * goal_y * (discount_rate ** i)
                gap.set_score(score)
            if (gap.score > max_score):
                max_score = gap.score
                best_gap = gap
            '''
            dist = np.linalg.norm([goal_x - gap.x, goal_y - gap.y])
            if (dist < min_dist):
                min_dist = dist
                best_gap = gap
            '''
        intermedia_goal = np.vstack([best_gap.x, best_gap.y, 0, 0])
        return intermedia_goal