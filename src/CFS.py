from abc import ABC, abstractmethod
from operator import concat
import pwd
from matplotlib.pyplot import axis
import numpy as np
from cvxopt import matrix, solvers
import cvxopt
import pdb
import timeit
import copy
from scipy.stats import multivariate_normal
from utils import *

K_EPSILON = 5.9915

class Planner(ABC):
    def __init__(self, spec, model) -> None:
        self.spec = spec
        self.model = model
        self.replanning_cycle = spec["replanning_cycle"]
        self.horizon = spec["horizon"]
        self._state_dimension = spec["state_dimension"]
        self.dT = spec["dT"]
    
    @property
    def state_dimension(self):
        return self._state_dimension

    @abstractmethod
    def _plan(self, dt: float, traj: np.array, est_data: dict) -> np.array:
        '''
            Implementation of planner
        '''
        pass

    def __call__(self, dt: float, traj: np.array, est_data: dict, transfer_function: matrix) -> np.array:
        '''
            Public interface
        '''
        return self._plan(dt, traj, est_data, transfer_function)

class CFSPlanner(Planner):

    def __init__(self, spec, model, uncertainty, safety_dist = 0.05) -> None:
        super().__init__(spec, model)
        self.max_speed = 0.02
        self.max_acc = 0.04
        self.has_uncertainty = uncertainty
        self.D = safety_dist
        self.Tcs = []

    def _CFS(self, 
        x_ref,
        n_ob,
        obs_traj,
        obs_uncertainty_dist,
        cq = [10,0,10], 
        cs = [0,1,0.1], 
        minimal_dis = 0, 
        ts = 1, 
        maxIter = 30,
        stop_eps = 1e-3
    ):
        # has obstacle, the normal CFS procedure 
        x_rs = np.array(x_ref)

        # planning parameters 
        h = x_rs.shape[0]    
        dimension = x_rs.shape[1]

        # flatten the trajectory to one dimension
        # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
        x_rs = np.reshape(x_rs, (x_rs.size, 1))
        x_origin = x_rs
        
        # objective terms 
        # identity
        Q1 = np.identity(h * dimension)
        S1 = Q1
        # velocity term 
        Vdiff = np.identity(h*dimension) - np.diag(np.ones((1,(h-1)*dimension))[0],dimension)
        Q2 = np.matmul(Vdiff.transpose(),Vdiff) 
        # Acceleration term 
        Adiff = Vdiff - np.diag(np.ones((1,(h-1)*dimension))[0],dimension) + np.diag(np.ones((1,(h-2)*dimension))[0],dimension*2)
        Q3 = np.matmul(Adiff.transpose(),Adiff)
        # Vdiff = eye(nstep*dim)-diag(ones(1,(nstep-1)*dim),dim);
        #pdb.set_trace()
        # objective 
        Q = Q1*cq[0]+Q2*cq[1]+Q3*cq[2]
        S = S1*cs[0]+Q2*cs[1]+Q3*cs[2]

        # quadratic term
        H =  Q + S 
        # linear term
        f = -1 * np.dot(Q, x_origin)

        b = np.ones((h * n_ob, 1)) * (-minimal_dis)
        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')
        # b = matrix(b,(len(b),1),'d')

        # reference trajctory cost 
        J0 =  np.dot(np.transpose(x_rs - x_origin), np.dot(Q, (x_rs - x_origin))) + np.dot(np.transpose(x_rs), np.dot(S, x_rs))
        J = float('inf')
        dlt = float('inf')
        cnt = 0

        # equality constraints 
        # start pos and end pos remain unchanged 
        Aeq = np.zeros((dimension*2, len(x_rs)))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i-1, len(x_rs)-i-1] = 1
        
        beq = np.zeros((dimension*2, 1))
        beq[0:dimension,0] = x_rs[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_rs[dimension*(h-1): dimension*h, 0] 
        # transform to convex optimization matrix 
        Aeq_array = Aeq
        beq_array = beq
        Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        beq = matrix(beq,(len(beq),1),'d')

        # main CFS loop
        Lstack, Sstack = [], []
        # inequality constraints 
        # l * x <= s
        Constraint = np.zeros((h * n_ob, len(x_rs)))
        
        for i in range(h):
            # get reference pos at time step i
            if i < h-1 and i > 0:
                x_r = x_rs[i * dimension : (i + 1) * dimension] 

                # get inequality value (distance)
                # get obstacle at this time step 
                for j in range(int(len(obs_traj[i])/2)):
                    obs_p = obs_traj[i,j*2:(j+1)*2]                      
                    dist = self._ineq(x_r,obs_p)
                    # get gradient 
                    ref_grad = jac_num(self._ineq, x_r, obs_p)
                    # compute
                    s = dist - self.D - np.dot(ref_grad, x_r)
                    if (self.has_uncertainty):
                        s -= obs_uncertainty_dist[i, j] #i * 0.001 #obs_uncertainty_dist[i, j] #
                    l = -1 * ref_grad
                    # update 
                    Sstack = vstack_wrapper(Sstack, s)
                    l_tmp = np.zeros((1, len(x_rs)))
                    l_tmp[:,i*dimension:(i+1)*dimension] = l
                    Lstack = vstack_wrapper(Lstack, l_tmp)
            if i == h-1 or i == 0: 
                s = np.zeros((1,1))
                l = np.zeros((1,2))

        # QP solver 
        Lstack = matrix(Lstack,(len(Lstack),len(Lstack[0])),'d')
        cvxopt.solvers.options['show_progress'] = False            
        while True:
            try:
                Sstack_matrix = matrix(Sstack,(len(Sstack),1),'d')
                sol = solvers.qp(H, f, Lstack, Sstack_matrix, Aeq, beq)
                x_ts = sol['x']
                break
            except ValueError:
                # no solution, relax the constraint               
                for i in range(len(Sstack)):
                    Sstack[i][0] += 0.01                        
            except ArithmeticError:
                print(f"Sstack {Sstack}, cnt {cnt}, maxIter {maxIter}")
                pdb.set_trace()
        
        x_ts = np.reshape(x_ts, (len(x_rs),1))
        J = np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + np.dot(np.transpose(x_ts), np.dot(S, x_ts))
        dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
        J0 = J
        x_rs = x_ts
        
        # return the reference trajectory      
        x_rs = x_rs[: h * dimension]
        x_rs = x_rs.reshape(h, dimension)
        return x_rs, J0

    def _ineq(self, x, obs):
        '''
        inequality constraints. 
        constraints: ineq(x) > 0
        '''
        # norm distance restriction
        obs_p = obs.flatten()
        obs_r = 0.03 
        obs_r = np.array(obs_r)
        
        # flatten the input x 
        x = x.flatten()
        dist = np.linalg.norm(x - obs_p) - obs_r

        return dist


    def _plan(self, dt: float, traj: np.array, est_data: dict, transfer_function: matrix) -> np.array:
        
        xd = self.state_dimension
        N = len(traj) 
        obs_pos_list = []
        obs_vel_list = []
        obs_cov_list = []
        obs_traj = []
        obs_uncertainty_dist = []
        F = transfer_function
        replan_step = 10
        for name, info in est_data['obstacle_sensor_est'].items():
            if 'obs' in name:
                obs_pos_list.append(info['pos'] - est_data['cartesian_sensor_est']['pos'])
                obs_vel_list.append(info['vel'])
                obs_cov_list.append(info['covariance'])

        # Without obstacle, then collision free
        if (len(obs_pos_list) == 0):
            return traj, 0, replan_step

        for obs_pos, obs_vel, obs_cov in zip(obs_pos_list, obs_vel_list, obs_cov_list):
            one_traj = []
            one_uncertainty_dist = []
            covariance = copy.deepcopy(obs_cov)
            for i in range(1,N+1):
                obs_waypoint = obs_pos + obs_vel * i * dt                
                obs_waypoint = obs_waypoint.reshape(1,-1).tolist()[0]
                one_traj.append(obs_waypoint) # [N, xd]
                # estimate safety dist that can guarantee safety in high probability                 
                covariance = F * covariance * F.transpose()  
                val, _ = np.linalg.eig(np.array(covariance.value)[[0,3]][:,[0,3]])
                safety_dist = 0
                for k in range(len(val)):
                    safety_dist += np.sqrt(K_EPSILON * val[k])
                safety_dist = safety_dist*0.1
                if (safety_dist > 0.03):
                    replan_step = min(i, replan_step)
                    safety_dist = min(0.03, safety_dist)
                one_uncertainty_dist.append(safety_dist)
            obs_traj.append(one_traj)
            obs_uncertainty_dist.append(one_uncertainty_dist)
        # min replan step is 5 to avoid replanning too often
        # because some agents have high covariance if they are sensed for the first time
        replan_step = max(replan_step, 5) 
        obs_traj = np.array(obs_traj)
        obs_uncertainty_dist = np.array(obs_uncertainty_dist)    
         
        if len(obs_traj) > 1:
            obs_traj = np.concatenate(obs_traj, axis=-1) # [T, n_obs * xd]
            obs_uncertainty_dist = obs_uncertainty_dist.T # [T, xd]
        else:
            obs_traj = obs_traj[0]
            obs_uncertainty_dist = obs_uncertainty_dist.reshape((N, 1))
        # CFS
        traj_pos_only = traj[:, :xd]
        traj_pos_safe, cost = self._CFS(x_ref=traj_pos_only, n_ob=len(obs_pos_list), obs_traj=obs_traj, obs_uncertainty_dist = obs_uncertainty_dist)
        traj[:, :xd] = traj_pos_safe       
        return traj, cost, replan_step



