from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.baxter import BaxterLeft, BaxterRight
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
# import warnings

from pyrep.backend import sim
import numpy as np
import math

PARENT_PATH = dirname(dirname(abspath(__file__)))
SCENE_FILE = join(PARENT_PATH, 'VrepEnviroment/baxter_final_env3.ttt')
# warnings.filterwarnings("ignore")


class ControlArm(object):
    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = BaxterLeft()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        
        self.OBS_NUM = 3
        self.obstracle = [Shape('Obs{}'.format(i)) for i in range(self.OBS_NUM)]
        self.obstracle_pos = [self.obstracle[i].get_position() for i in range(self.OBS_NUM)]
        self.obstracle_orien = [self.obstracle[i].get_orientation() for i in range(self.OBS_NUM)]
        self.target = Shape('Target')
        self.path_step = 0
        
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.POS_MIN = [0.6, 0.1, 1.0]
        self.POS_MAX = [1.2, 0.9, 1.2]

        # optional #
        self.path = None
        self.joint_pos = None
        self.prv_joint_pos = np.zeros(7) # need to be fix 
        self.joint_vel = None
        ############
        
        self.eps = 0.2
        # self.eps2 = 0.15
    
    def genPath(self):
        pos = self.target.get_position()
        self.path = self.agent.get_path(position=pos, euler=[0, math.radians(180), 0])

    def getCurrentUser(self,disturbance_enable = True):
        terminate = False
        offset = self.path._arm._num_joints * (self.path_step)
        disturbance = 3.14*(np.random.rand(self.path._arm._num_joints)-0.5)/5
        dt = sim.simGetSimulationTimeStep()
        
        if offset + len(self.path._arm.joints) >= len(self.path._path_points):
            try:
                self.genPath()
            except:
                terminate = True
            self.path_step = 0
        else:
            self.joint_pos = self.path._path_points[offset:offset + len(self.path._arm.joints)]
            if disturbance_enable:
                self.joint_pos += disturbance

            self.joint_vel = (self.prv_joint_pos - self.joint_pos)/dt

            rand = np.random.rand(1)[0]
            if rand <= 0.4 or not disturbance_enable:
                self.path_step += 1
        
            ax, ay, az = self.agent_ee_tip.get_position()
            tx, ty, tz = self.target.get_position()
            error = np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
            
            self.prv_joint_pos = self.joint_pos   
            if error < self.eps:
                terminate = True  
            else:
                terminate = False

        return self.joint_pos, self.joint_vel, terminate


    def getState(self):
        _ , _ , terminate = self.getCurrentUser()
        return np.concatenate([self.joint_pos,
                            self.joint_vel,
                            self.agent.get_joint_positions(),
                            self.agent.get_joint_velocities(),
                            self.target.get_position()]) , terminate
    def reset(self):
        # Get a random position within a cuboid and set the target position

        pos = list(np.random.uniform(self.POS_MIN, self.POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)

        for i in range(self.OBS_NUM):
            self.obstracle[i].set_position(self.obstracle_pos[i])
            self.obstracle[i].set_orientation(self.obstracle_orien[i])
        
        self.genPath()
        state,_ = self.getState()
        return state

    def step(self, action):
        self.agent.set_joint_target_velocities(3.14*0.4*action + 0.6*self.joint_pos)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        
        # Reward is negative distance to target
        error = np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        reward = -error

        done = True if error < self.eps else False
        state, terminate = self.getState()
        
        # if terminate and reward < 0:
        #     reward *= 3
        # print(reward)
        return state, reward, done, terminate
    
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()