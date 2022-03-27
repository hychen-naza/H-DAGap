import math
from math import sqrt, acos, atan2, sin, cos
import numpy as np
from operator import attrgetter
import pdb
import copy

class LaserData:
    def __init__(self, depth: float, theta: float, rel_x: float, rel_y: float, v_x: float, v_y: float, id = -1):
        self.x = rel_x
        self.y = rel_y
        self.vx = v_x
        self.vy = v_y
        self.depth = depth
        self.theta = theta
        self.id = id

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
        self.obs_radius = 0.03 # obs radius

    def update_obs_radius(self, new_radius):
        self.obs_radius = new_radius

    def calcu_angle(self, rel_x, rel_y):
        angle = math.atan2(rel_x, rel_y)
        angle = int(angle * 180 / math.pi)
        if (angle < 0):
            angle += 360
        return angle

    def calcu_tangent_point(self, obs_pos, robot_pos, radius, angle):
        # follow this stackoverflow https://stackoverflow.com/questions/49968720/find-tangent-points-in-a-circle-from-a-point
        Px, Py = robot_pos
        Cx, Cy = obs_pos

        b = sqrt((Px - Cx)**2 + (Py - Cy)**2)  # hypot() also works here
        try:
            th = acos(radius / b)  # angle theta
        except ValueError:
            #print(f"radius {radius}, b {b}")
            del self.inflated_depths[angle]
            return []
            #pdb.set_trace()
        d = atan2(Py - Cy, Px - Cx)  # direction angle of point P from C
        d1 = d + th  # direction angle of point T1 from C
        d2 = d - th  # direction angle of point T2 from C

        t1x = Cx + radius * cos(d1)
        t1y = Cy + radius * sin(d1)
        t1angle = math.atan2(t1x-Px, t1y-Py)
        t1angle = int(t1angle * 180 / math.pi)
        t2x = Cx + radius * cos(d2)
        t2y = Cy + radius * sin(d2)
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
        tangent_points = [ltp]
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
            # shink angle to the nearest y
            #angle = round(angle/90)*90
            if (angle not in self.depths):
                #print(f"angle {angle}")
                self.depths[angle] = []
            self.depths[angle].append(LaserData(depth, angle, rel_x, rel_y, obs_vel_state[0], obs_vel_state[1], obs_state['id']))
        # pick the closest one in every egocircle bucket
        tangent_points = []
        for angle, laser_datas in self.depths.items():           
            self.inflated_depths[angle] = min(laser_datas, key=attrgetter('depth'))
            #print(f"angle {angle}, self.inflated_depths[angle] {self.inflated_depths[angle]}")
            tangent_points.extend(self.calcu_tangent_point([self.inflated_depths[angle].x+robot_pos_state[0],self.inflated_depths[angle].y+robot_pos_state[1]], [robot_pos_state[0], robot_pos_state[1]], self.obs_radius, angle))
        return tangent_points
    
    def build_gaps(self, sensor_data: dict):
        robot_state = sensor_data['cartesian_sensor_est']['pos'].flatten()        
        angles = list(self.inflated_depths.keys())
        angles.sort()        
        gaps = []
        fake_obs_id = -1
        # no obstacle, move towards the goal
        if (len(angles) == 0):
            depth = 0.15              
            angles = [-45, 45]
            for fake_theta in angles:
                rel_x = depth * math.sin(fake_theta * 2 * math.pi / 360)
                rel_y = depth * math.cos(fake_theta * 2 * math.pi / 360)
                self.inflated_depths[fake_theta] = LaserData(depth, fake_theta, rel_x, rel_y,0,0,fake_obs_id)
                self.calcu_tangent_point([rel_x+robot_state[0],rel_y+robot_state[1]], [robot_state[0], robot_state[1]], self.obs_radius, fake_theta)                
                fake_obs_id -= 1
        # one obstacle
        elif (len(angles) == 1):
            laser_data = self.inflated_depths[angles[0]] 
            depth = 0.15           
            angle = 90
            num_gap = int(360 / angle)
            angles = [laser_data.theta]
            for i in range(1, num_gap):
                fake_theta = (laser_data.theta + i * angle) % 360
                angles.append(fake_theta)
                rel_x = depth * math.sin(fake_theta * 2 * math.pi / 360)
                rel_y = depth * math.cos(fake_theta * 2 * math.pi / 360)
                self.inflated_depths[fake_theta] = LaserData(depth, fake_theta, rel_x, rel_y, 0, 0,fake_obs_id)
                self.calcu_tangent_point([rel_x+robot_state[0],rel_y+robot_state[1]], [robot_state[0], robot_state[1]], self.obs_radius,fake_theta)
                fake_obs_id -= 1
        # multiple obstacles
        else:
            angle_start = copy.deepcopy(angles)
            angle_end = copy.deepcopy(angles[1:])
            angle_end.append(angles[0])
            for angle1, angle2 in zip(angle_start, angle_end):  
                # split the big empty chunk              
                angle = 90
                depth = 0.15   
                diff_angle = (angle2 - angle1) % 360
                num_gap = int(diff_angle / angle)
                if (num_gap >= 2):
                    for i in range(1, num_gap):
                        fake_theta = (angle1 + i*angle) % 360
                        angles.append(fake_theta)
                        rel_x = depth * math.sin(fake_theta * 2 * math.pi / 360)
                        rel_y = depth * math.cos(fake_theta * 2 * math.pi / 360)
                        self.inflated_depths[fake_theta] = LaserData(depth, fake_theta, rel_x, rel_y, 0, 0,fake_obs_id)
                        self.calcu_tangent_point([rel_x+robot_state[0],rel_y+robot_state[1]], [robot_state[0], robot_state[1]], self.obs_radius, fake_theta)
                        fake_obs_id -= 1                        
            angles.sort()  
        gaps = self.build_gap_from_angles(angles, robot_state)
        self.inflated_depths = {}
        self.inflated_tangent_points = {}
        self.depths = {}
        return gaps
    
    def build_gap_from_angles(self, angles, robot_state):
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
        angle_end = {}
        for i in range(len(angles)):
            lobs_angle = self.inflated_tangent_points[angles[i]]['rta']
            lobs_rpoint = self.inflated_tangent_points[angles[i]]['rtp']
            lobs_lpoint = self.inflated_tangent_points[angles[i]]['ltp']
            min_id = -1
            min_diff = float("inf")
            for j in range(len(angles)):
                robs_angle = self.inflated_tangent_points[angles[j]]['lta']
                robs_lpoint = self.inflated_tangent_points[angles[j]]['ltp']    
                robs_rpoint = self.inflated_tangent_points[angles[j]]['rtp']                    
                diff_angle = min((robs_angle - lobs_angle) % 360, (lobs_angle - robs_angle) % 360)
                lr2rl_dist = np.linalg.norm([lobs_rpoint[0] - robs_lpoint[0], lobs_rpoint[1] - robs_lpoint[1]])
                ll2rr_dist = np.linalg.norm([lobs_lpoint[0] - robs_rpoint[0], lobs_lpoint[1] - robs_rpoint[1]])
                if (i != j and diff_angle < min_diff and lr2rl_dist < ll2rr_dist):
                    min_id = j
                    min_diff = diff_angle
            #angle_end.append(angles[min_id])
            if (angles[min_id] not in angle_end):
                angle_end[angles[min_id]] = []
            angle_end[angles[min_id]].append(angles[i])
        #print(f"angle_end {angle_end}")
        min_angle_end = {}
        for end, starts in angle_end.items():
            min_diff = float("inf")
            robs_angle = self.inflated_tangent_points[end]['lta']
            for start in starts:
                lobs_angle = self.inflated_tangent_points[start]['rta']
                diff_angle = min((robs_angle - lobs_angle) % 360, (lobs_angle - robs_angle) % 360)
                if (diff_angle < min_diff):
                    min_diff = diff_angle
                    min_angle_end[end] = start

        angle_start = [v for _, v in min_angle_end.items()]
        angle_end = [k for k, _ in min_angle_end.items()]
        if (len(angle_start) == 0):            
            pdb.set_trace()
            angle_start = angles
            angle_end = angles[1:]
            angle_end.append(angles[0])
        #print(f"angle_start {angle_start}, angle_end {angle_end}")
        #angle_end = angles[1:]
        #angle_end.append(angles[0])
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
                #print("here")
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
            #pdb.set_trace()
        try:
            possible_gap_ids = [gap.id for gap in gaps]
        except:
            pdb.set_trace()
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