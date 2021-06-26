import sim
import rospy
import sys
from geometry_msgs.msg import Pose, Quaternion, Point, Twist, Vector3
import itertools


class SimHandle:

    DOCKER_IP = '192.168.65.2'

    def __init__(self):
        sim.simxFinish(-1)
        self.clientID = sim.simxStart(self.DOCKER_IP, 8080, True, True, 5000, 5)
        if self.clientID == -1:
            rospy.logerr('Failed connecting to remote API server')
            sim.simxFinish(-1)
            sys.exit(1)
        rospy.loginfo('Connected to remote API server')
        rospy.loginfo('Testing connection')
        objs = self.run_sim_function(sim.simxGetObjects, (self.clientID, sim.sim_handle_all, sim.simx_opmode_blocking))
        rospy.loginfo(f'Number of objects in the scene: {len(objs)}')
        self.robot = self.run_sim_function(sim.simxGetObjectHandle, (self.clientID, "Rob", sim.simx_opmode_blocking))
        self.set_position_to_zero()
        rospy.sleep(0.1)
        self.init_streaming()
        rospy.loginfo("Starting main loop")

    def init_streaming(self):
        self.get_pose(mode=sim.simx_opmode_streaming)
        self.get_twist(mode=sim.simx_opmode_streaming)

    def run_sim_function(self, func, args):
        res = func(*args)
        if not isinstance(res, list) and not isinstance(res, tuple):
            res = (res,)
        if res[0] != sim.simx_return_ok and args[-1] != sim.simx_opmode_streaming:
            rospy.logerr(f'Error calling simulation. Code: {res[0]}')
        if len(res) == 1:
            return None
        if len(res) == 2:
            return res[1]
        return res[1:]

    def set_position_to_zero(self):
        self.run_sim_function(sim.simxSetObjectPosition, (self.clientID, self.robot, -
                              1, [0.0, 0.0, 0.0], sim.simx_opmode_blocking))

    def set_thruster_force(self, force):
        inp = itertools.chain.from_iterable(force)
        self.run_sim_function(sim.simxCallScriptFunction, (self.clientID, "Rob", sim.sim_scripttype_childscript,
                                                           "setThrusterForces",
                                                           [], list(inp), [""], bytearray(),
                                                           sim.simx_opmode_blocking))

    def get_mass(self):
        out = self.run_sim_function(sim.simxCallScriptFunction, (self.clientID, "Rob", sim.sim_scripttype_childscript,
                                                                 "getMass",
                                                                 [self.robot], [], [""], bytearray(),
                                                                 sim.simx_opmode_blocking))
        return out[1][0]

    def get_pose(self, mode=sim.simx_opmode_buffer):
        pos = self.run_sim_function(sim.simxGetObjectPosition, (self.clientID, self.robot, -1, mode))
        quat = self.run_sim_function(sim.simxGetObjectQuaternion, (self.clientID, self.robot, -1, mode))
        return Pose(position=Point(*pos), orientation=Quaternion(*quat))

    def get_twist(self, mode=sim.simx_opmode_buffer):
        lin, ang = self.run_sim_function(sim.simxGetObjectVelocity, (self.clientID, self.robot, mode))
        return Twist(linear=Vector3(*lin), angular=Vector3(*ang))
