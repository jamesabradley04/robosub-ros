#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from custom_msgs.msg import ThrusterSpeeds
import numpy as np
from thruster_manager import ThrusterManager
from std_srvs.srv import SetBool
import controls_utils as utils
import resource_retriever as rr


class ThrusterController:
    """ROS node that manages thruster allocation and publishing once PID loops have generated
    control efforts. Also manages power control if PID loops are bypassed.

    Attributes:
        enable_service: The ROS Service used for enabling/disabling thruster publishing (soft e-stop)
        enabled: Whether controls is enabled
        MAX_THRUSTER_POWER: The maximum thruster power expected by downstream packages (currently 8-bit signed int)
        pid_outputs: The desired control efforts in local frame (generated by PID loops)
        POWER_SCALING_FACTOR: The factor by which to scale the thruster outputs for safety. Must be between [0, 1]
        powers: The desired powers in local frame (used in place of PID control efforts in power control)
        ROBOT_PUB_TOPIC: The topic that thruster allocations get published to
        RUN_LOOP_RATE: The rate at which thruster allocations are published
        t_allocs: The Tx1 vector of thruster allocations ranging from [-1, 1]
        thruster_speeds_pub: The ROS publisher that publishes 8-bit thruster speeds used by downstream packages
        tm: The ThrusterManager object used to calculate thruster allocations
    """

    ROBOT_PUB_TOPIC = '/offboard/thruster_speeds'
    RUN_LOOP_RATE = 10  # 10 Hz
    MAX_THRUSTER_POWER = 127  # Some nuance, max neg power is -128. Ignoring that for now
    POWER_SCALING_FACTOR = 0.5

    def __init__(self):
        """Initializes the ROS node, creating thruster manager and required pub/sub configuration.
        Controls is disabled by default and requires a service call to enable output.
        """
        rospy.init_node('thruster_controls')

        self.thruster_speeds_pub = rospy.Publisher(self.ROBOT_PUB_TOPIC, ThrusterSpeeds, queue_size=3)
        self.enable_service = rospy.Service('enable_controls', SetBool, self._handle_enable_controls)

        self.tm = ThrusterManager(rr.get_filename('package://controls/config/cthulhu.config', use_protocol=False))

        self.enabled = False
        self.pid_outputs = np.zeros(6)
        self.powers = np.zeros(6)
        self.t_allocs = np.zeros(8)

        for d in utils.get_axes():
            rospy.Subscriber(utils.get_controls_move_topic(d), Float64, self._on_pid_received, d)
            rospy.Subscriber(utils.get_power_topic(d), Float64, self._on_power_received, d)

    def _handle_enable_controls(self, req):
        """Handles requests to the enable ROS service, disabling/enabling output accordingly. An example call is
        `rosservice call /enable_controls true`.

        Args:
            req: The request data sent in the service call. In this case, a boolean denoting whether to enable.

        Returns:
            A message relaying the enablement status of controls.
        """
        self.enabled = req.data
        return {'success': True, 'message': 'Successfully set enabled to ' + str(req.data)}

    def _on_pid_received(self, val, direction):
        """Callback that stores PID control efforts for use in the run loop. Also updates thruster allocations based
        on new control effort.

        TODO: Examine performance and determine if the thruster calculation should be done in the run loop or as part
        of this PID callback. Theoretically these calculations only have to be done once per run loop (before
        publishing to off-board comms). PID loops are publishing with min frequency of 100 Hz, so we are recalculating
        thruster allocations at 600 Hz. We can cut this down to 10 Hz.

        Args:
            val: The PID control effort (float ranging from [-1, 1])
            direction: The axis the control effort maps to
        """
        self.pid_outputs[utils.get_axes().index(direction)] = val.data
        self.t_allocs = self.tm.calc_t_allocs(self.pid_outputs)

    def _on_power_received(self, val, direction):
        """Callback that stores powers for use in the run loop. This is used for power control, which bypasses PID
        control efforts. If the power in a direction is 0, PID loops are not bypassed. This enables stabilization on
        axes that don't have a power setpoint.

        TODO: Same as _on_pid_received

        Args:
            val: The desired power (float ranging from [-1, 1])
            direction: The axis the desired power maps to
        """
        if val.data != 0:
            self.pid_outputs[utils.get_axes().index(direction)] = val.data
        self.t_allocs = self.tm.calc_t_allocs(self.pid_outputs)

    def run(self):
        """Loop that publishes thruster allocations to corresponding topic. If disabled, zeroes are published to make
        sure thrusters don't spin.
        """
        rate = Rospy.rate(self.RUN_LOOP_RATE)
        while not rospy.is_shutdown():
            if not self.enabled:
                i8_t_allocs = ThrusterSpeeds()
                i8_t_allocs.speeds = np.zeros(8).astype(int)
                self.thruster_speeds_pub.publish(i8_t_allocs)

            if self.enabled:
                self._scale_thruster_speeds()
                i8_t_allocs = ThrusterSpeeds()
                i8_t_allocs.speeds = (self.t_allocs * self.MAX_THRUSTER_POWER * self.POWER_SCALING_FACTOR).astype(int)
                self.thruster_speeds_pub.publish(i8_t_allocs)

            rate.sleep()

    def _scale_thruster_speeds(self):
        """Scales thruster speeds according to a custom algorithm. Doesn't scale if all allocations are 0.

        TODO: Determine if our scaling algorithm is valid. Currently, we first scale the max thruster allocation to the
        max PID control effort. Then we clamp values between -1 and 1 and multiply by max thruster speed.
        """
        t_alloc_max = float(np.max(np.absolute(self.t_allocs)))
        pid_max = float(np.max(np.absolute(self.pid_outputs)))

        if t_alloc_max != 0:
            self.t_allocs *= pid_max / t_alloc_max

        # Clamp values of t_allocs to between -1 to 1. Added for safety but probably not required.
        self.t_allocs = np.clip(self.t_allocs, -1, 1)


def main():
    try:
        ThrusterController().run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
