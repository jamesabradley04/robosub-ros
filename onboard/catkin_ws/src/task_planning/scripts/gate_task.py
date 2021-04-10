SIDE_THRESHOLD = 0.1  # means gate post is within 1 tenth of the side of the frame
CENTERED_THRESHOLD = 0.1  # means gate will be considered centered if within 1 tenth of the center of the frame

def create_gate_task_sm(velocity=0.2):
    sm = smach.StateMachine(outcomes=['succeeded', 'failed'])

    with sm:
        smach.StateMachine.add('NEARGATE', NearGateTask(SIDE_THRESHOLD),
                                    transitions={'true':'MOVETHROUGHGATE', 'false':'HORIZONTALALIGNMENT'})
        smach.StateMachine.add('MOVETHROUGHGATE', MoveToPoseLocalTask(3, 0, 0, 0, 0, 0),
                                    transitions={'done':'succeeded'})
        smach.StateMachine.add('HORIZONTALALIGNMENT', GateVerticalAlignmentTask(CENTERED_THRESHOLD),
                                    transitions={'left':'ROTATELEFT', 'right':'ROTATERIGHT', 'center':'VERTICALALIGNMENT'})
        smach.StateMachine.add('ROTATELEFT', AllocateVelocityLocalTask(0, 0, 0, 0, 0, self.velocity),
                                    transitions={'done':'HORIZONTALALIGNMENT'})
        smach.StateMachine.add('ROTATERIGHT', AllocateVelocityLocalTask(0, 0, 0, 0, 0, -self.velocity),
                                    transitions={'done':'HORIZONTALALIGNMENT'})
        smach.StateMachine.add('VERTICALALIGNMENT', GateVerticalAlignmentTask(CENTERED_THRESHOLD),
                                    transitions={'top':'ASCEND', 'bottom':'DESCEND', 'center':'ADVANCE'})
        smach.StateMachine.add('ASCEND', AllocateVelocityLocalTask(0, 0, self.velocity, 0, 0, 0),
                                    transitions={'done':'VERTICALALIGNMENT'})
        smach.StateMachine.add('DESCEND', AllocateVelocityLocalTask(0, 0, -self.velocity, 0, 0, 0),
                                    transitions={'done':'VERTICALALIGNMENT'})
        smach.StateMachine.add('ADVANCE', AllocateVelocityLocalTask(self.velocity, 0, 0, 0, 0, 0),
                                    transitions={'done':'NEARGATE'})

    return sm

def scrutinize_gate(self, gate_data, gate_tick_data):
    """Finds the distance from the gate to each of the four edges of the frame
    Parameters:
    gate_data (custom_msgs/CVObject): cv data for the gate
    gate_tick_data (custom_msgs/CVObject): cv data for the gate tick
    Returns:
    dict: left - distance from left edge of gate to frame edge (from 0 to 1)
            right - distance from right edge of gate to frame edge (from 0 to 1)
            top - distance from top edge of gate to frame edge (from 0 to 1)
            bottom - distance from bottom edge of gate to frame edge (from 0 to 1)
            offset_h - difference between distances on the right and left sides (from 0 to 1)
            offset_v - difference between distances on the top and bottom sides (from 0 to 1)
    """
    if gate_data.label == 'none':
        return None
    res = {}
    res["left"] = gate_data.xmin
    res["right"] = 1 - gate_data.xmax
    res["top"] = gate_data.ymin
    res["bottom"] = 1 - gate_data.ymax

    # Adjust the target area if the gate tick is detected
    if gate_tick_data.label != 'none' and gate_tick_data.score > 0.5:
        # If the tick is closer to the left side
        if abs(gate_tick_data.xmin - gate_data.xmin) < abs(gate_tick_data.xmax - gate_data.xmax):
            res["right"] = 1 - gate_tick_data.xmax
        else:
            res["left"] = gate_tick_data.xmin
        
        res["bottom"] = 1 - gate_tick_data.ymax

    res["offset_h"] = res["right"] - res["left"]
    res["offset_v"] = res["bottom"] - res["top"]

    return res