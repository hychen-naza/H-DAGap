from builtins import range
from builtins import object
import math
import numpy as np
import random
import copy
import pdb
import time
import random

SUCCESS = 'success'
FAILURE_TOO_MANY_STEPS = 'too_many_steps'

# Custom failure states for navigation.
NAV_FAILURE_COLLISION = 'collision'
NAV_FAILURE_OUT_OF_BOUNDS = 'out_of_bounds'

def l2( xy0, xy1 ):
    ox = xy1[0]
    oy = xy1[1]
    dx = xy0[0] - xy1[0]
    dy = xy0[1] - xy1[1]
    dist = math.sqrt( (dx * dx) + (dy * dy) )
    if (xy1[0] < -0.9):
        warp_dx = xy0[0] - (1 + (xy1[0] + 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (1 + (xy1[0] + 1))
            dist = dist1
    elif (xy1[0] > 0.9):
        warp_dx = xy0[0] - (-1 + (xy1[0] - 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (-1 + (xy1[0] - 1))
            dist = dist1
    return dist, ox, oy



class Env(object):
	def __init__(self, display, field, robot_state,
                    min_dist,
                    noise_sigma,
                    in_bounds,
                    goal_bounds,
                    nsteps, is_online):

		self.init_robot_state = copy.deepcopy(robot_state)
		self.robot_state = copy.deepcopy(self.init_robot_state)

		self.field = field
		self.display = display

		self.min_dist = min_dist
		self.in_bounds = in_bounds
		self.goal_bounds = goal_bounds
		self.nsteps = 3500#nsteps
		self.cur_step = 0
		self.dt = 0.1
		self.cfs_tc = 0
		self.is_online = is_online
		self.collision_time = 0

	def reset(self):
		self.cur_step = 0
		self.cfs_tc = 0
		self.robot_state = copy.deepcopy(self.init_robot_state)
		self.display.setup( self.field.x_bounds, self.field.y_bounds,
							self.in_bounds, self.goal_bounds,
							margin = self.min_dist)
		self.field.random_init() # randomize the init position of obstacles
		cx,cy,_ = self.robot_state.position
		obstacle_id, obstacle_pos, _ = self.find_nearest_obstacle(cx,cy)
		state = [cx, cy, self.robot_state.v_x, self.robot_state.v_y]
		relative_pos = [cx - obstacle_pos[0], cy - obstacle_pos[1]]
		return np.array(state + relative_pos)

	def find_nearest_obstacle(self, cx, cy, unsafe_obstacle_ids = []):
		t = self.cur_step * self.dt
		if (self.is_online):
			t += self.cfs_tc
		astlocs = self.field.obstacle_locations(t, cx, cy, self.min_dist * 5)
		nearest_obstacle = None
		nearest_obstacle_id = -1
		nearest_obstacle_dist = np.float("inf")    
		collisions = ()
		for i,x,y in astlocs:
			self.display.obstacle_at_loc(i,x,y)
			if (i in unsafe_obstacle_ids):
				self.display.obstacle_set_color(i, 'blue')
			dist, ox, oy = l2( (cx,cy), (x,y) )
			if dist < self.min_dist:
				collisions += (i,)
			if dist < nearest_obstacle_dist:
				nearest_obstacle_dist = dist
				nearest_obstacle = [ox, oy, 0, 0]
				nearest_obstacle_id = i
		if (nearest_obstacle_id == -1):
			nearest_obstacle = [-1, -1, 0, 0]
		return nearest_obstacle_id, nearest_obstacle, collisions

	def display_start(self):
		self.display.begin_time_step(self.cur_step)

	def display_end(self):
		self.display.end_time_step(self.cur_step)

	def clean(self):
		self.display.clean()

	def save_env(self):
		self.cur_step_copy = self.cur_step
		self.robot_state_copy = copy.deepcopy(self.robot_state)
		self.field_copy = copy.deepcopy(self.field)
		return

	def read_env(self):
		self.cur_step = self.cur_step_copy
		self.robot_state = self.robot_state_copy
		self.field = self.field_copy
		return 

	def step(self, dt, action, is_safe = False, unsafe_obstacle_ids = []):
		'''
		action: [dv_x, dv_y]
		'''
		self.cur_step += 1
		self.robot_state = self.robot_state.steer(dt, action[0][0], action[1][0] )
		cx,cy,ch = self.robot_state.position
		#
		self.display.robot_at_loc( cx, cy, ch, is_safe)
		nearest_obstacle_id, nearest_obstacle, collisions = self.find_nearest_obstacle(cx, cy, unsafe_obstacle_ids)

		next_robot_state = [cx, cy, self.robot_state.v_x, self.robot_state.v_y]
		relative_pos = [cx - nearest_obstacle[0], cy - nearest_obstacle[1]]
		next_state = next_robot_state + relative_pos

		# done
		done = False
		arrive = False
		reward_wo_cost = 0
		if collisions:
			self.collision_time += 1
			ret = (NAV_FAILURE_COLLISION, self.cur_step)
			self.display.navigation_done(*ret)
			done = True
			reward = -500
		elif self.goal_bounds.contains( (cx,cy) ):
			ret = (SUCCESS, self.cur_step)
			self.display.navigation_done(*ret)
			done = True
			reward = 2000
			arrive = True
		elif self.cur_step > self.nsteps:
			done = True
			reward = 0
		else:
			# rewards depend on current state
			relative_dist = np.sqrt(relative_pos[0]**2 + relative_pos[1]**2)
			reward = 0 
			reward_wo_cost = 0 #(cy+1)
		info = {'arrive':arrive, 'reward_wo_cost':reward_wo_cost}
		return np.array(next_state), reward, done, info

	def find_unsafe_obstacles(self, min_dist):
		cx, cy, _ = self.robot_state.position
		t = self.cur_step * self.dt		
		if (self.is_online):
			t += self.cfs_tc
		unsafe_obstacles = self.field.unsafe_obstacle_locations(t, cx, cy, min_dist)
		unsafe_obstacle_ids = [ele[0] for ele in unsafe_obstacles]
		unsafe_obstacle_info = [np.array(ele[1]) for ele in unsafe_obstacles]
		return unsafe_obstacle_ids, unsafe_obstacle_info

	def online_update(self, cfs_time, last_robot_action):
		self.cfs_tc += cfs_time
		self.robot_state = self.robot_state.steer(cfs_time, last_robot_action[0][0], last_robot_action[1][0] )