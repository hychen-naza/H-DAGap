from __future__ import print_function
from __future__ import absolute_import

# python modules
import argparse
import numpy as np
import time
import threading

# project files
import dynamic_obstacle
import bounds
import robot # double integrator robot
import simu_env
import runner
import param
from turtle_display import TurtleRunnerDisplay
from SSA import SafeSetAlgorithm
from CFS import CFSPlanner
from GlobalPlanner import GlobalPlanner
from DynamicModel import DoubleIntegrator 
from Controller import NaiveFeedbackController
from InflatedEgoCircle import InflatedEgoCircle
from KalmanFilter import KalmanFilter, F 


def display_for_name( dname ):
    # choose none display or visual display
    if dname == 'turtle':
        return TurtleRunnerDisplay(800,800)
    else:
        return runner.BaseRunnerDisplay()

dT = 0.1
is_online = True
obs_num = 50

def run_kwargs( params ):
    in_bounds = bounds.BoundsRectangle( **params['in_bounds'] )
    goal_bounds = bounds.BoundsRectangle( **params['goal_bounds'] )
    min_dist = params['min_dist']
    ret = { 'field': dynamic_obstacle.ObstacleField(dt = dT, obs_num = obs_num),
            'robot_state': robot.DoubleIntegratorRobot( **( params['initial_robot_state'] ) ),
            'in_bounds': in_bounds,
            'goal_bounds': goal_bounds,
            'noise_sigma': params['noise_sigma'],
            'min_dist': min_dist,
            'nsteps': 1000,
            'is_online': is_online }
    return ret

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( '--display',
                       choices=('turtle','text','none'),
                       default='none' )
    prsr.set_defaults(feature=False)
    return prsr

replan = True
cur_state = None
cur_sensor_data = None
planning_init_state = None
replanning_timer = 0
replanning_cycle = 10
safe_traj = None

class PlannerThread (threading.Thread):
    def __init__(self, global_planner):
      threading.Thread.__init__(self)
      self.global_planner = global_planner

    def run(self):
      global replan
      global planning_init_state
      global safe_traj
      global replanning_timer
      global replanning_cycle
      while (True):
        if replan:
          safe_traj, estimate_replan_cycle = self.global_planner.MultiDynTrajPlanner(cur_state[:4], cur_sensor_data)
          replanning_cycle = estimate_replan_cycle
          planning_init_state = cur_state[:2] 
          replanning_timer = 0 
          replan = False
        else:
          time.sleep(0)

def main(display_name):
    # testing env
    try:
        params = param.params
    except Exception as e:
        print(e)
        return
    display = display_for_name(display_name)
    env_params = run_kwargs(params)

    # parameters
    max_steps = int(1e6)
    episode_reward = 0
    episode_num = 0
    total_steps = 0
    # dynamic model parameters
    fx = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])
    print(display_name)
    
    collision_num = 0
    failure_num = 0
    success_num = 0
    # Goal position
    goal_pos = np.array([episode_num/100, 1.05])
    # Parameters
    horizon = 30
    global_horizon = 60
    global replanning_cycle
    global replanning_timer
    global cur_sensor_data
    global cur_state
    global replan

    replanning_timer = replanning_cycle
    # Build the env
    env = simu_env.Env(display, **(env_params))
    sensor_dist = 0.2
    # Init Safe Controller SSA
    safe_controller = SafeSetAlgorithm(max_speed = env.robot_state.max_speed, dmin = 0.1)
    
    # Init CFS Planner
    dynamic_model = DoubleIntegrator({'dT': dT})
    cfs_planner = CFSPlanner({'horizon': horizon, 'replanning_cycle': replanning_cycle, 'state_dimension': 2, 'dT':dT}, dynamic_model, True)
    # Init Feedback Controller for CFS
    controller = NaiveFeedbackController()
    # Init EgoCircle
    inflated_ego_circle = InflatedEgoCircle(goal_pos, dist = 0.1, radius = sensor_dist, safety_dist = 0.05)
    # Init Global Planner
    global_planner = GlobalPlanner(dist = 0.1, global_dist = 0.25, horizon = horizon, global_horizon = global_horizon, \
                                    model = dynamic_model, cfs_planner = cfs_planner, egocircle = inflated_ego_circle, F = F, has_uncertainty = True)
    # Init KalmanFilter
    kf_estimator = KalmanFilter()


    state, done = env.reset(), False
    infeasible_traj_count = 0
    plannerThread = PlannerThread(global_planner)
    
    robot_traj_state = []
    robot_traj_state.append(state[:2])
    for t in range(max_steps):     
      env.display_start()

      # Collect robot information and obstacles information
      # use estimated position and velocity information instead of the ground truth value
      sensor_data = {}
      sensor_data['cartesian_sensor_est'] = {'pos':np.vstack(state[:2]), 'vel':np.vstack(state[2:4])}
      unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(sensor_dist)
      obstacle_locations = []
      for id, obs in zip(unsafe_obstacle_ids, unsafe_obstacles):
        measure_uncertainty = np.minimum(np.maximum(np.random.normal(0, 0.01, 2), -0.01), 0.01)
        obstacle_locations.append([id, obs[0]+measure_uncertainty[0], obs[1]+measure_uncertainty[1]])
      kf_estimator.observe_obstacles(obstacle_locations)
      sensor_data['obstacle_sensor_est'] = {}   
      for i, (id, obs) in enumerate(zip(unsafe_obstacle_ids, unsafe_obstacles)):
        estimated_info = kf_estimator.obstacle_measure[id].estimate
        sensor_data['obstacle_sensor_est']['obs'+str(i)] = {'pos':np.vstack([estimated_info[0][0], estimated_info[3][0]]), 'vel':np.vstack([estimated_info[1][0], estimated_info[4][0]]), 
                                                            'covariance': kf_estimator.obstacle_measure[id].covariance, 'id':id}
      
      cur_sensor_data = sensor_data
      cur_state = state
      if (t == 0):
        plannerThread.start()
      
      # wait until we have init traj
      if (env.cur_step == 0):
        while (safe_traj is None):
          time.sleep(0)

      # High level planner and CFS planner      
      if replanning_timer == replanning_cycle:
        replan = True     
        
      # Generate control signal  
      sensor_data['planning_init_state'] = np.vstack(planning_init_state)      
      next_traj_point = safe_traj[min(replanning_timer, safe_traj.shape[0]-1)]
      next_traj_point = np.vstack(next_traj_point.ravel())      
      action = controller(dT, sensor_data, next_traj_point, 2)
      replanning_timer += 1 
      for i in range(len(safe_traj) - 2):
          cur_waypoint = safe_traj[i]
          next_waypoint = safe_traj[i+1]
          if (abs(cur_waypoint[0] - next_waypoint[0]) > 0.03 or abs(cur_waypoint[1] - next_waypoint[1]) > 0.03):
            infeasible_traj_count += 1
            break

      # Monitor and modify the unsafe control signal
      action, is_safe = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
      robot_traj_state.append(state[:2])
      action = np.vstack(action.ravel())      
      s_new, reward, done, info = env.step(dT, action, is_safe, unsafe_obstacle_ids) 
      episode_reward += reward        
      env.display_end()
      state = s_new
      kf_estimator.estimate_obstacle_locs()

      if (done and reward == 0):      
        failure_num += 1    
      elif (done and reward == 2000):
        # display the trajectory and save the image
        #env.display.traj_at_loc(robot_traj_state, [0,0])
        #env.display.save_eps(t)
        success_num += 1
        total_steps += env.cur_step
      elif (done and reward < 2000):
        collision_num += 1   
      
      if (done):              
        print(f"Train: episode_num {episode_num}, total_steps {total_steps}, reward {episode_reward}, collision {env.collision_time}") 
        episode_reward = 0
        episode_num += 1
        state, done = env.reset(), False
        kf_estimator = KalmanFilter()
        if (episode_num >= 100):
          try:
            print(f"avg_succ_steps {total_steps/success_num}, collision_num {collision_num}, success_num {success_num}, failure_num {failure_num}")
          except ZeroDivisionError:
            print(f"collision_num {collision_num}, success_num {success_num}, failure_num {failure_num}")
          break


if __name__ == '__main__':
    main(display_name = "turtle")
      



