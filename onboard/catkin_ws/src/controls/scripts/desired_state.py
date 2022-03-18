#!/usr/bin/env python3

from turtle import position
import rospy
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64, Bool
import controls_utils as utils
from tf import TransformListener


class bcolors:
    BOLD = '\033[1m'
    OKGREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'


class DesiredStateHandler:
    DESIRED_TWIST_TOPIC = 'controls/desired_twist'
    DESIRED_POSE_TOPIC = 'controls/desired_pose'
    DESIRED_POWER_TOPIC = 'controls/desired_power'

    REFRESH_HZ = 10  # for main loop

    # All variables are a dictionary with mappings between the strings in DIRECTIONS to its corresponding value
    pose = None  # Desired pose
    twist = None  # Desired twist
    power = None  # Desired power
    position_mode = {} # Whether we are controlling position or velocity
    # These dictionaries contain mappings between the strings in DIRECTIONS to the corresponding rospy publisher objects
    pub_pos = {}
    pub_pos_enable = {}
    pub_vel = {}
    pub_vel_enable = {}
    pub_control_effort = {}
    pub_power = {}

    def __init__(self):
        rospy.init_node('desired_state')

        for d in utils.get_axes():
            self.pub_pos[d] = rospy.Publisher(utils.get_pid_topic(d), Float64, queue_size=3)
            self.pub_pos_enable[d] = rospy.Publisher(utils.get_pos_pid_enable(d), Bool, queue_size=3)
            self.pub_vel_enable[d] = rospy.Publisher(utils.get_vel_pid_enable(d), Bool, queue_size=3)
            self.pub_vel[d] = rospy.Publisher(utils.get_vel_topic(d), Float64, queue_size=3)
            self.pub_control_effort[d] = rospy.Publisher(utils.get_controls_move_topic(d), Float64, queue_size=3)
            self.pub_power[d] = rospy.Publisher(utils.get_power_topic(d), Float64, queue_size=3)
            rospy.Subscriber(utils.get_position_mode_topic(d), Bool, lambda val : self._on_position_mode_received(axis, val))

        self.listener = TransformListener()

        rospy.Subscriber(self.DESIRED_POSE_TOPIC, Pose, self._on_pose_received)
        rospy.Subscriber(self.DESIRED_TWIST_TOPIC, Twist, self._on_twist_received)
        rospy.Subscriber(self.DESIRED_POWER_TOPIC, Twist, self._on_power_received)

    def _on_pose_received(self, pose):
        self.pose = utils.parse_pose(utils.transform_pose(self.listener, 'odom', 'base_link', pose))

    def _on_twist_received(self, twist):
        self.twist = utils.parse_twist(twist)

    def _on_power_received(self, power):
        self.power = utils.parse_twist(power)

    def _on_position_mode_received(self, axis, is_position):
        self.position_mode[axis] = is_position

    def soft_estop(self):
        # Stop Moving
        self.disable_loops()
        utils.publish_data_constant(self.pub_control_effort, 0)
        self.twist = None
        self.pose = None
        self.power = None

    def disable_loops(self):
        utils.publish_data_constant(self.pub_pos_enable, False)
        utils.publish_data_constant(self.pub_vel_enable, False)

    def enable_loops(self):
        # Enable all PID Loops
        utils.publish_data_constant(self.pub_pos_enable, True)
        utils.publish_data_constant(self.pub_vel_enable, True)

    def set_position_modes(self, modes):
        # Enable all PID Loops that are in position mode
        for axis, mode in modes.items():
            self.pub_pos_enable[axis].publish(mode)

    def disable_pos_loop(self):
        # Disable position loop
        utils.publish_data_constant(self.pub_pos_enable, False)

    def run(self):
        rate = rospy.Rate(self.REFRESH_HZ)

        warned = False
        event_id = 0

        while not rospy.is_shutdown():
            rate.sleep()

            if (self.pose and self.power) or (self.twist and self.power):
                # More than one seen in one update cycle, so warn and continue
                rospy.logerr("===> Controls received conflicting desired states! Halting robot. <===")
                self.soft_estop()
                continue
            elif not self.pose and not self.twist and not self.power:
                self.soft_estop()
                if not warned:
                    rospy.logwarn(bcolors.WARN + ("===> Controls received no desired state! Halting robot. "
                                                  "(Event %d) <===" % event_id) + bcolors.RESET)
                    warned = True
                continue
            elif self.power is None and ((any(self.position_mode) and self.pose is None) or (not all(self.position_mode) and self.twist is None)):
                self.soft_estop()
                if not warned:
                    rospy.logwarn(bcolors.WARN + ("===> Controls received incomplete desired state! Halting robot. "
                                                  "(Event %d) <===" % event_id) + bcolors.RESET)
                    warned = True
                continue

            # Now we have either pose/twist XOR powers
            if warned:
                rospy.loginfo(bcolors.OKGREEN + ("===> Controls now receiving desired state (End event %d) <===" %
                                                 (event_id)) + bcolors.RESET)
                event_id += 1
                warned = False

            if self.pose:
                # Run both loops
                self.set_position_modes(self.position_mode)
                utils.publish_data_dictionary(self.pub_pos, self.pose)
                for axis, mode in self.position_mode.items():
                    if not mode:
                        self.pub_vel[axis].publish(self.twist[axis])
                self.pose = None

            elif self.twist:
                # Just run velocity loop
                self.enable_loops()
                self.disable_pos_loop()
                utils.publish_data_dictionary(self.pub_vel, self.twist)
                self.twist = None

            elif self.power:
                self.disable_loops()
                # Enable stabilization on all axes with 0 power input
                for p in self.power.keys():
                    if self.power[p] == 0:
                        utils.publish_data_constant(self.pub_vel_enable, True)
                # Publish velocity setpoints to all Velocity loops, even though some are not enabled
                utils.publish_data_dictionary(self.pub_vel, self.power)
                utils.publish_data_dictionary(self.pub_power, self.power)
                self.power = None


def main():
    try:
        DesiredStateHandler().run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
