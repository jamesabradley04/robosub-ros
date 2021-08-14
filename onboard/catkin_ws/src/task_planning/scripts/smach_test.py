#!/usr/bin/env python

import rospy
import smach
from move_tasks import MoveToPoseGlobalTask

# define state Foo

# main
def main():
    rospy.init_node('smach_test')

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['finish'])

    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('Move', MoveToPoseGlobalTask(), 
                               transitions={'spin':'Move', 
                                            'done':'finish'})

    # Execute SMACH plan
    outcome = sm.execute()


if __name__ == '__main__':
    main()