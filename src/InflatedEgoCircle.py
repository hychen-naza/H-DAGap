import math
from math import sqrt, acos, atan2, sin, cos
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
        obs_id = 'obs'+str(self.id)
        if (obs_id not in sensor_data['obstacle_sensor_est']):
            return 0
        covariance = sensor_data['obstacle_sensor_est'][obs_id]['covariance']
        val, _ = np.linalg.eig(np.array(covariance.value)[:2,:2])
        uncertainty_bound = 0
        for k in range(len(val)):
            uncertainty_bound += np.sqrt(K_EPSILON * val[k])
        return min(0.03, uncertainty_bound*0.1)

class Gap:
    def __init__(self, x, y, ltp, rtp, id):
        # the coordinate of mid_point
        self.x = x 
        self.y = y
        # the left and right obstacles
        self.ltp = ltp
        self.rtp = rtp
        self.id = id #str(left_laserdata.id) + '_' + str(right_laserdata.id)
    
    def set_score(self, score):
        self.score = score

    def update(self, dt):
        self.left_obs.x = self.left_obs.x + dt * self.left_obs.vx
        self.left_obs.y = self.left_obs.y + dt * self.left_obs.vy
        self.right_obs.x = self.right_obs.x + dt * self.right_obs.vx
        self.right_obs.y = self.right_obs.y + dt * self.right_obs.vy


class InflatedEgoCircle:
    def __init__(self, goal: list, dist: float, radius: float, safety_dist: float):
        self.depths = {}
        self.inflated_depths = {}     
        self.inflated_tangent_points = {}   
        self.goal = np.array(goal)
        self.dist = dist # intermedia goal dist
        self.radius = radius # laser sensing range
        self.safety_dist = safety_dist
        self.obs_radius = 0.05 # obs radius
        self.angle_interval = 90

    #def update_obs_radius(self, new_radius):
    #    self.obs_radius = new_radius

    def calcu_angle(self, rel_x, rel_y):
        angle = math.atan2(rel_x, rel_y)
        angle = int(angle * 180 / math.pi)
        if (angle < 0):
            angle += 360
        return angle

    def calcu_tangent_point(self, obs_pos, robot_pos, radius, angle, sensor_data):
        # follow this stackoverflow https://stackoverflow.com/questions/49968720/find-tangent-points-in-a-circle-from-a-point
        Px, Py = robot_pos
        Cx, Cy = obs_pos

        uncertainty_bound = self.inflated_depths[angle].calcu_uncertainty_bound(sensor_data)
        robust_radius = radius + uncertainty_bound

        b = sqrt((Px - Cx)**2 + (Py - Cy)**2)  # hypot() also works here
        try:
            th = acos(robust_radius / b)  # angle theta
        except ValueError:
            del self.inflated_depths[angle]
            return []

        d = atan2(Py - Cy, Px - Cx)  # direction angle of point P from C
        d1 = d + th  # direction angle of point T1 from C
        d2 = d - th  # direction angle of point T2 from C

        t1x = Cx + robust_radius * cos(d1)
        t1y = Cy + robust_radius * sin(d1)
        t1angle = math.atan2(t1x-Px, t1y-Py)
        t1angle = int(t1angle * 180 / math.pi)
        t2x = Cx + robust_radius * cos(d2)
        t2y = Cy + robust_radius * sin(d2)
        t2angle = math.atan2(t2x-Px, t2y-Py)
        t2angle = int(t2angle * 180 / math.pi)
        
        if (t1angle * t2angle <= 0):
            # check the -0 to 0 boundary cross
            if (abs(t1angle) < 90 and abs(t2angle) < 90):
                if (t1angle > t2angle):
                    rtp = [t1x-Px, t1y-Py]
                    rta = t1angle
                    ltp = [t2x-Px, t2y-Py]
                    lta = t2angle + 360
                else:
                    ltp = [t1x-Px, t1y-Py]
                    lta = t1angle + 360
                    rtp = [t2x-Px, t2y-Py]
                    rta = t2angle
            # check the 180 to -180 boundary cross
            else:
                if (t1angle > t2angle):
                    ltp = [t1x-Px, t1y-Py]
                    lta = t1angle
                    rtp = [t2x-Px, t2y-Py]
                    rta = t2angle + 360                    
                else:
                    rtp = [t1x-Px, t1y-Py]
                    rta = t1angle + 360
                    ltp = [t2x-Px, t2y-Py]
                    lta = t2angle
        else:
            if (t2angle < 0):
                t2angle += 360
            if (t1angle < 0):
                t1angle += 360
            if (t1angle > t2angle):
                ltp = [t2x-Px, t2y-Py]
                lta = t2angle
                rtp = [t1x-Px, t1y-Py]
                rta = t1angle
            else:
                ltp = [t1x-Px, t1y-Py]
                lta = t1angle
                rtp = [t2x-Px, t2y-Py]
                rta = t2angle
        tangent_points = [ltp,[Cx,Cy]]
        #print(f"in calcu_tangent_point angle {angle} ltp {ltp}, lta {lta}, rtp {rtp}, rta {rta}")
        self.inflated_tangent_points[angle] = {'ltp':ltp,'lta':lta,'rtp':rtp,'rta':rta}
        return tangent_points

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
        tangent_points = []
        for angle, laser_datas in self.depths.items():           
            self.inflated_depths[angle] = min(laser_datas, key=attrgetter('depth'))
            self.calcu_tangent_point([self.inflated_depths[angle].x+robot_pos_state[0],self.inflated_depths[angle].y+robot_pos_state[1]], [robot_pos_state[0], robot_pos_state[1]], self.obs_radius, angle, sensor_data)
        return tangent_points
    
    def build_gaps(self, sensor_data: dict):
        robot_state = sensor_data['cartesian_sensor_est']['pos'].flatten()        
        angles = list(self.inflated_depths.keys())
        angles.sort()        
        gaps = []
        fake_obs_id = -1
        # no obstacle, move towards the goal
        if (len(angles) <= 1):
            depth = self.radius              
            angles = [-45, 45]
            for fake_theta in angles:
                rel_x = depth * math.sin(fake_theta * 2 * math.pi / 360)
                rel_y = depth * math.cos(fake_theta * 2 * math.pi / 360)
                self.inflated_depths[fake_theta] = LaserData(depth, fake_theta, rel_x, rel_y,0,0,fake_obs_id)
                self.calcu_tangent_point([rel_x+robot_state[0],rel_y+robot_state[1]], [robot_state[0], robot_state[1]], self.obs_radius, fake_theta, sensor_data)                
                fake_obs_id -= 1
        # one obstacle
        elif (len(angles) == 1):
            laser_data = self.inflated_depths[angles[0]]         
            num_gap = int(360 / self.angle_interval)
            angles = [laser_data.theta]
            for i in range(1, num_gap):
                fake_theta = (laser_data.theta + i * self.angle_interval) % 360
                angles.append(fake_theta)
                rel_x = self.radius * math.sin(fake_theta * 2 * math.pi / 360)
                rel_y = self.radius * math.cos(fake_theta * 2 * math.pi / 360)
                self.inflated_depths[fake_theta] = LaserData(self.radius, fake_theta, rel_x, rel_y, 0, 0,fake_obs_id)
                self.calcu_tangent_point([rel_x+robot_state[0],rel_y+robot_state[1]], [robot_state[0], robot_state[1]], self.obs_radius,fake_theta, sensor_data)
                fake_obs_id -= 1
        # multiple obstacles
        else:
            angle_start = copy.deepcopy(angles)
            angle_end = copy.deepcopy(angles[1:])
            angle_end.append(angles[0])
            for angle1, angle2 in zip(angle_start, angle_end):  
                # split the big empty chunk              
                diff_angle = (angle2 - angle1) % 360
                num_gap = int(diff_angle / self.angle_interval)
                if (num_gap >= 2):
                    for i in range(1, num_gap):
                        fake_theta = (angle1 + i*self.angle_interval) % 360
                        angles.append(fake_theta)
                        rel_x = self.radius * math.sin(fake_theta * 2 * math.pi / 360)
                        rel_y = self.radius * math.cos(fake_theta * 2 * math.pi / 360)
                        self.inflated_depths[fake_theta] = LaserData(self.radius, fake_theta, rel_x, rel_y, 0, 0,fake_obs_id)
                        self.calcu_tangent_point([rel_x+robot_state[0],rel_y+robot_state[1]], [robot_state[0], robot_state[1]], self.obs_radius, fake_theta, sensor_data)
                        fake_obs_id -= 1                        
            angles.sort()  
        gaps = self.build_gap_from_angles(angles, robot_state, sensor_data)
        self.inflated_depths = {}
        self.inflated_tangent_points = {}
        self.depths = {}
        return gaps
    
    def build_gap_from_angles(self, angles, robot_state, sensor_data):
        '''
        lobs_angles = {}
        for angle in angles:
            lobs_angle = self.inflated_tangent_points[angle]['lta']
            lobs_angles[angle] = lobs_angle
        angle_start = [k for k, _ in sorted(lobs_angles.items(), key=lambda item: item[1])]
        angle_end = angles[1:]
        angle_end.append(angles[0])
        '''
        angle_start = angles
        angle_end = angles[1:]
        angle_end.append(angles[0])
        gaps = []
        best_gap = None
        max_dist = float("-inf")
        for angle1, angle2 in zip(angle_start, angle_end):                
            laser_data1 = self.inflated_depths[angle1]
            laser_data2 = self.inflated_depths[angle2]
            gap_id = str(laser_data1.id) + '_' + str(laser_data2.id) 
            # check there is a gap using the distance between centers of inflated obs
            dist = np.linalg.norm([laser_data1.x - laser_data2.x,laser_data1.y - laser_data2.y])
            lobs_angle = self.inflated_tangent_points[angle1]['rta']
            lobs_point = self.inflated_tangent_points[angle1]['rtp']
            robs_angle = self.inflated_tangent_points[angle2]['lta']
            robs_point = self.inflated_tangent_points[angle2]['ltp']            
            diff_angle = min((robs_angle - lobs_angle) % 360, (lobs_angle - robs_angle) % 360)
            #print(f"angle1 {lobs_angle}, angle2 {robs_angle}, dist {dist}, ltp {self.inflated_tangent_points[angle1]['rtp']}, rtp {self.inflated_tangent_points[angle2]['ltp']}") 
            if (dist > max_dist):
                max_dist = dist
                avg_depth = (laser_data1.depth + laser_data2.depth) / 2
                mid_angle = (lobs_angle + diff_angle / 2) % 360   
                rel_x = avg_depth * math.sin(mid_angle * 2 * math.pi / 360)
                rel_y = avg_depth * math.cos(mid_angle * 2 * math.pi / 360)  
                best_gap = Gap(rel_x, rel_y, self.inflated_tangent_points[angle1]['rtp'], self.inflated_tangent_points[angle2]['ltp'], gap_id)
                
            if (dist > 2 * self.obs_radius and diff_angle > 30): #(robs_angle > lobs_angle or (robs_angle < lobs_angle and lobs_angle>180 and robs_angle<180))): # cross boundary, but we have 180 degree buffer                
                # not the -0 to 0 boundary cross case
                if (robs_angle - lobs_angle < 0 and robs_angle - lobs_angle > -180):
                    continue
                avg_depth = (laser_data1.depth + laser_data2.depth) / 2
                mid_angle = (lobs_angle + diff_angle / 2) % 360   
                rel_x = avg_depth * math.sin(mid_angle * 2 * math.pi / 360)
                rel_y = avg_depth * math.cos(mid_angle * 2 * math.pi / 360)                 
                gaps.append(Gap(rel_x, rel_y, self.inflated_tangent_points[angle1]['rtp'], self.inflated_tangent_points[angle2]['ltp'], gap_id))
            
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