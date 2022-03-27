import sys, os
import numpy as np
from abc import ABC, abstractmethod
import pdb
class FeedbackController(ABC):
    '''
        Feedback Controller Base
    '''

    def __init__(self, is_global):
        self.is_global = is_global

    @abstractmethod
    def _control(self, processed_data: dict, error: np.ndarray) -> np.ndarray:
        '''
            Can be model inverse or other control algo
        '''
        pass
    
    def __call__(self,
        dt: float,
        sensors_data: dict,
        goal: np.ndarray,
        state_dimension: int
    ) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''

        # goal -> control space error (pos + vel)
        # e.g., for unicycle, convert x/y coord (planned goal) to error in vel/heading        
        if (self.is_global):
            # control based on the global frame 
            e_pos = sensors_data['cartesian_sensor_est']['pos'] - goal[:state_dimension]
            e_vel = sensors_data['cartesian_sensor_est']['vel'] - goal[state_dimension:state_dimension+state_dimension]
        else:
            # control based on the robot frame 
            e_pos = (sensors_data['cartesian_sensor_est']['pos'] - sensors_data['planning_init_state']) - goal[:state_dimension] 
            e_vel = sensors_data['cartesian_sensor_est']['vel'] - goal[state_dimension:state_dimension+state_dimension]         
        e = np.concatenate((e_pos, e_vel))
        # control space error -> action
        # e.g., for unicycle, compute vel/heading from vel/heading error
        u = self._control(processed_data=sensors_data, error=e)

        return u


class NaiveFeedbackController(FeedbackController):

    def __init__(self, is_global = False):
        super().__init__(is_global)

        # weights
        self.kp = 1 #spec["kp"]
        self.kv = 0.05#1#0.5#0.05 #spec["kv"]
        self.u_max = 0.04 #spec["u_max"]

    def _control(self, processed_data: dict, error: np.ndarray) -> np.ndarray:
        '''
            P control on both pos and vel
            Then use control model to convert to action
        '''
        #super()._control(processed_data, error)
        n = error.shape[0]
        assert(n % 2 == 0)
        # compute u as desired state time derivative for control model
        u = -self.kp*error[:n//2] - self.kv * error[n//2:]
        for i in range(u.shape[0]):
            u[i] = np.clip(u[i], -self.u_max, self.u_max)

        return u
