import numpy as np
import pdb
import math
import copy
import collections
import time

class GlobalPlanner:
    def __init__(self, dist, global_dist, horizon, global_horizon, model, cfs_planner, egocircle, F, has_uncertainty):
        self.dist = dist # the distance between intermedia waypoints
        self.global_dist = global_dist
        self.horizon = horizon # the planning horizon
        self.global_horizon = global_horizon
        self.model = model # dynamic model of the robot
        self._state_dimension = 2
        self.cfs_planner = cfs_planner
        self.ego_circle = egocircle
        self.F = F
        self.has_uncertainty = has_uncertainty
        self.diff = 0
        self.top_two_diff = 0
        self.top_three_diff = 0
        self.traj_num = []

    def IntermediaGoalPlanner(self, dt: float, robot_state: np.ndarray, goal: np.ndarray, sensor_data: dict):
        '''
        Return:
            intermedia_goal: np.array, []
            traj: np.array, trajectory from the current robot state to the intermedia goal
        '''        
        goal_rel_pos = goal - np.array(robot_state[:2])
        goal_rel_pos = self.dist * goal_rel_pos / (np.linalg.norm(goal_rel_pos, 2))        
        intermedia_goal = np.vstack(np.array([goal_rel_pos[0], goal_rel_pos[1], 0, 0]))
        traj = self.IntegratorPlanner(dt, robot_state, intermedia_goal, self.horizon)
        return intermedia_goal, traj


    def MultiDynTrajPlanner(self, robot_state: np.ndarray, sensor_data, dt = 0.1):
        
        gap_trajs = {}
        while(1):
            try:
                init_sensor_data = copy.deepcopy(sensor_data)
                break
            except RuntimeError:
                pass
        self.ego_circle.parse_sensor_data(sensor_data)
        simple_path_plan = False
        if (len(list(self.ego_circle.inflated_depths.keys())) < 1):
            init_pos = np.array(robot_state[:2])
            goal_pos = np.array([0.35, 0.95])
            vel_direction = goal_pos - init_pos
            vel_direction = vel_direction / np.linalg.norm(vel_direction)
            gap_trajs['0_0'] = []
            for step in range(30):
                gap_trajs['0_0'].append(list(vel_direction * 0.02 * step * dt))
            simple_path_plan = True
        dgap_start = time.time()
        if (not simple_path_plan):
            for i in range(30):
                self.ego_circle.parse_sensor_data(sensor_data)
                # build gaps
                possible_gaps = self.ego_circle.build_gaps(sensor_data)
                if (i == 0):
                    for gap in possible_gaps:
                        gap_trajs[gap.id] = [[0,0]]

                for gap in possible_gaps:                
                    if (gap.id in gap_trajs): #and not gap_passed_trajs[gap.id]                    
                        cur_pos = self.calcu_gap_traj(dt, gap, gap_trajs[gap.id][-1])
                        gap_trajs[gap.id].append(cur_pos)
                    elif (gap.id not in gap_trajs):
                        # generate a new topology for that new gap
                        min_dist = float("inf")
                        min_gap_id = None
                        for gap_id in gap_trajs.copy():
                            if (gap.id != gap_id and np.linalg.norm([gap_trajs[gap_id][-1][0]-gap.x, gap_trajs[gap_id][-1][1]-gap.y]) < min_dist):
                                min_dist = np.linalg.norm([gap_trajs[gap_id][-1][0]-gap.x, gap_trajs[gap_id][-1][1]-gap.y])
                                min_gap_id = gap_id
                        if (min_gap_id is None):
                            gap_trajs[gap.id] = []
                        else:
                            gap_trajs[gap.id] = copy.deepcopy(gap_trajs[min_gap_id])                 
                        # calculate 
                        cur_pos = self.calcu_gap_traj(dt, gap, gap_trajs[gap.id][-1])
                        gap_trajs[gap.id].append(cur_pos)

                # update sensor data based on estimation
                for obs_id, _ in sensor_data['obstacle_sensor_est'].items(): #enumerate(zip(unsafe_obstacle_ids, unsafe_obstacles)):
                    sensor_data['obstacle_sensor_est'][obs_id]['pos'] = sensor_data['obstacle_sensor_est'][obs_id]['pos'] + sensor_data['obstacle_sensor_est'][obs_id]['vel'] * dt
        safe_trajs_ids = []
        score = {}
        cfs_score = {}
        replan_times = {}
        safe_trajs = {}
        for k, v in gap_trajs.items():
            if (len(v) > 10):
                safe_trajs_ids.append(k)
                score[k] = v[-1][1]
        if (not safe_trajs_ids):
            pdb.set_trace()
        score = {k:v for k, v in sorted(score.items(), key=lambda item: item[1])}
        score_ids = [k for k, _ in sorted(score.items(), key=lambda item: item[1])]
        safe_trajs_ids = score_ids[-2:]

        cfs_start = time.time()
        for id in safe_trajs_ids:
            traj = np.array(gap_trajs[id][1:])
            for _ in range(1):
                traj_vel = (traj[1:,:2] - traj[:-1,:2]) * (1./dt)
                traj_vel = np.vstack((traj_vel, traj_vel[-1]))
                traj = np.hstack((traj[:,:2], traj_vel))
                safe_traj = traj
                cfs_score[id] = safe_traj[-1][1]
                safe_traj, cost, replan_time = self.cfs_planner(dt, traj, init_sensor_data, self.F) 
                traj = safe_traj 
            goal_dist = np.linalg.norm([0.35-safe_traj[-1][0], 0.95-safe_traj[-1][1]])
            cost = 0
            cfs_score[id] = -cost - goal_dist 
            safe_trajs[id] = safe_traj
            replan_times[id] = 5 #replan_time

        best_traj_id = max(cfs_score, key=cfs_score.get)
        return safe_trajs[best_traj_id], replan_times[best_traj_id]

    def MultiStaticTrajPlanner(self, robot_state: np.ndarray, sensor_data, dt = 0.1):
        score = {}
        safe_trajs = {}
        self.ego_circle.parse_sensor_data(sensor_data)
        simple_path_plan = False
        if (len(list(self.ego_circle.inflated_depths.keys())) < 1):
            init_pos = np.array(robot_state[:2])
            goal_pos = np.array([0.35, 0.95])
            vel_direction = goal_pos - init_pos
            vel_direction = vel_direction / np.linalg.norm(vel_direction)
            traj = []
            for step in range(30):
                traj.append(list(vel_direction * 0.02 * step * dt))
            traj = np.array(traj)
            traj_vel = (traj[1:] - traj[:-1]) * (1./dt)
            traj_vel = np.vstack((traj_vel, traj_vel[-1]))
            traj = np.hstack((traj, traj_vel))
            score['0_0'] = 0            
            safe_trajs['0_0'] = traj
            simple_path_plan = True
        # build gaps
        if (not simple_path_plan):
            possible_gaps = self.ego_circle.build_gaps(sensor_data)
            for gap in possible_gaps:
                traj = self.IntegratorPlanner(dt, robot_state, self.horizon, gap)
                traj_vel = (traj[1:] - traj[:-1]) * (1./dt)
                traj_vel = np.vstack((traj_vel, traj_vel[-1]))
                traj = np.hstack((traj, traj_vel))
                safe_traj = traj
                score[gap.id] = -np.linalg.norm([0.35-safe_traj[-1][0], 0.95-safe_traj[-1][1]]) #safe_traj[-1][1]
                safe_trajs[gap.id] = safe_traj
        best_traj_id = max(score, key=score.get)
        return safe_trajs[best_traj_id], 5

    def LinearIntegratorPlanner(self, dt, cur_state, goal_state, N):
        # assume integrater uses first _state_dimension elements from est data
        state = np.vstack(np.array([0, 0, cur_state[2], cur_state[3]])) 
        xd = self._state_dimension
        # both state and goal is in [pos, vel, etc.]' with shape [T, ?, 1]
        A = self.model.A(dt=dt)
        B = self.model.B(dt=dt)

        # lifted system for tracking last state
        Abar = np.vstack([np.linalg.matrix_power(A, i) for i in range(1,N+1)])
        Bbar = np.vstack([
            np.hstack([
                np.hstack([np.linalg.matrix_power(A, p) @ B for p in range(row, -1, -1)]),
                np.zeros((xd, N-1-row))
            ]) for row in range(N)
        ])
        # tracking each state dim
        n_state_comp = 2
        traj = np.zeros((N, xd * n_state_comp, 1))
        for i in range(xd):
            # vector: pos, vel, etc. of a single dimension
            x = np.vstack([ state[ j * xd + i, 0 ] for j in range(n_state_comp) ])
            xref = np.vstack([ goal_state[ j * xd + i, 0 ] for j in range(n_state_comp) ])

            ubar = np.linalg.lstsq(
                a = Bbar[-xd:, :], b = xref - np.linalg.matrix_power(A, N) @ x)[0] # get solution

            xbar = (Abar @ x + Bbar @ ubar).reshape(N, n_state_comp, 1)

            for j in range(n_state_comp):
                traj[:, j * xd + i] = xbar[:, j]
        traj = traj.squeeze()    
        return traj

    def IntermediaGoalPlannerv2(self, dt: float, robot_state: np.ndarray, best_gap):
        traj = self.IntegratorPlanner(dt, robot_state, self.horizon, best_gap)
        return traj

    def IntegratorPlanner(self, dt, cur_state, N, best_gap):
        # assume integrater uses first _state_dimension elements from est data
        dgap = copy.deepcopy(best_gap)
        state = np.vstack(np.array([0, 0, cur_state[2], cur_state[3]])) 
        xd = self._state_dimension
        # both state and goal is in [pos, vel, etc.]' with shape [T, ?, 1]
        A = self.model.A(dt=dt)
        B = self.model.B(dt=dt)
        n_state_comp = 2 
        traj = [[0,0]] 
        for i in range(30):
            traj.append(self.calcu_gap_traj(dt, dgap, traj[-1]))
        traj = np.array(traj[1:])
        traj_vel = (traj[1:] - traj[:-1]) * (1./dt)
        traj_vel = np.vstack((traj_vel, traj_vel[-1]))
        traj = np.hstack((traj, traj_vel))
        return traj

    def calcu_gap_traj(self, dt, gap, cur_pos): 
        # goal gradient
        goal_pos = np.array([gap.x, gap.y])
        cur_pos = copy.deepcopy(cur_pos)
        rel_goal = goal_pos - cur_pos              
        rg = np.linalg.norm(rel_goal)
        thetax = math.atan2(rel_goal[0], rel_goal[1])
        rel_goal_vec = (rg * math.sin(thetax), rg * math.cos(thetax))
        result = (rel_goal_vec / (np.linalg.norm(rel_goal_vec)))            
        # obstacle rotation gradient, all these are relative position
        close_obs = []
        rot_angle = math.pi / 2
        _sigma = 1
        lobs = np.array([gap.ltp[0] - cur_pos[0], gap.ltp[1] - cur_pos[1]])
        robs = np.array([gap.rtp[0] - cur_pos[0], gap.rtp[1] - cur_pos[1]])
        r_pi2 = np.array([[math.cos(rot_angle), -math.sin(rot_angle)], [math.sin(rot_angle), math.cos(rot_angle)]])
        reg_r_pi2 = np.array([[math.cos(-rot_angle), -math.sin(-rot_angle)], [math.sin(-rot_angle), math.cos(-rot_angle)]])
        c1 = np.array([0,0])
        c2 = np.array([0,0])
        c1 = -1 * r_pi2 @ (lobs / np.linalg.norm(lobs)) * np.exp(-10*np.linalg.norm(lobs) / _sigma)
        c2 = -1 * reg_r_pi2 @ (robs / np.linalg.norm(robs)) * np.exp(-10*np.linalg.norm(robs) / _sigma) 
        result += (c1 + c2) 
        # modify the result to make it meet dynamic limitation
        max_v = 0.03
        result = dt * max_v * (result / np.linalg.norm(result))
        cur_pos = cur_pos + np.array(result)
        return list(cur_pos)
