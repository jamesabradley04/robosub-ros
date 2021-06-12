from move_tasks import MoveToPoseGlobalTask, AllocateVelocityLocalTask, MoveToPoseLocalTask
from combination_tasks import IndSimulTask, DepSimulTask, LeaderFollowerTask, ListTask
from task import Task
import task_utils
import numpy as np
from geometry_msgs.msg import Point

class GateTask(Task):
    SIDE_THRESHOLD = 0.1  # means gate post is within 1 tenth of the side of the frame
    CENTERED_THRESHOLD = 0.1  # means gate will be considered centered if within 1 tenth of the center of the frame

    def __init__(self, velocity=0.2):
        super(GateTask, self).__init__()
        self.velocity = velocity

    def _on_task_start(self):
        self.gate_search_condition = NearGateTask(self.SIDE_THRESHOLD)  # finishes when can see gate and close to gate
        self.rotate_condition = GateCenteredTask(self.CENTERED_THRESHOLD)  # finishes when gate is centered horizontally
        self.rotate = LeaderFollowerTask(self.rotate_condition, AllocateVelocityLocalTask(0, 0, 0, 0, 0, self.velocity))
        self.descend_condition = GateInDirectionTask(self.CENTERED_THRESHOLD, "bottom", return_when_true=False) # finishes when gate is no longer below the robot
        self.descend = LeaderFollowerTask(self.descend_condition, AllocateVelocityLocalTask(0, 0, -self.velocity, 0, 0, 0))
        self.ascend_condition = GateInDirectionTask(self.CENTERED_THRESHOLD, "top", return_when_true=False) # finishes when gate is no longer above the robot
        self.ascend = LeaderFollowerTask(self.ascend_condition, AllocateVelocityLocalTask(0, 0, self.velocity, 0, 0, 0))
        self.advance_condition = GateCenteredTask(self.CENTERED_THRESHOLD, return_on_center=False)  # finishes when/if gate is not centered
        self.advance = LeaderFollowerTask(self.advance_condition, AllocateVelocityLocalTask(self.velocity, 0, 0, 0, 0, 0))
        self.gate_magic = ListTask([
            LeaderFollowerTask(self.gate_search_condition, ListTask([self.rotate, self.descend, self.ascend, self.advance], -1)),
            MoveToPoseLocalTask(3, 0, 0, 0, 0, 0)])

    def _on_task_run(self):
        self.gate_magic.run()
        if self.gate_magic.finished:
            self.finish()


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


class NearGateTask(Task):
    def __init__(self, threshold):
        super(NearGateTask, self).__init__()
        self.threshold = threshold

    def _on_task_run(self):
        gate_info = scrutinize_gate(self.cv_data['gate'], self.cv_data['gate_tick'])
        if gate_info:
            if (gate_info["left"] > self.threshold) and (gate_info["right"] > self.threshold):
                self.finish()


# Returns when the gate appears towards the direction specified
class GateInDirectionTask(Task):
    def __init__(self, threshold, direction, return_when_true=True):
        super(GateInDirectionTask, self).__init__()
        self.threshold = threshold
        # direction should be "left", "right", "top", or "bottom"
        self.direction = direction.lower()
        self.return_when_true = return_when_true

    def _on_task_run(self):
        gate_info = scrutinize_gate(self.cv_data['gate'], self.cv_data['gate_tick'])
        if gate_info:
            offset = gate_info["offset_h" if direction in ["left", "right"] else "offset_v"]
            if abs(offset) > self.threshold and (offset < 0 if direction in ["right", "bottom"] else offset > 0):
                if self.return_when_true:
                    self.finish()
            elif not(return_when_true):
                self.finish()


class GateCenteredTask(Task):
    def __init__(self, threshold, return_on_center=True, horizontal=True):
        super(GateCenteredTask, self).__init__()
        self.threshold = threshold
        self.return_on_center = return_on_center
        self.horizontal = horizontal

    def _on_task_run(self):
        gate_info = scrutinize_gate(self.cv_data['gate'], self.cv_data['gate_tick'])
        if gate_info:
            if self.return_on_center:
                if abs(gate_info["offset_h" if horizontal else "offset_v"]) < threshold:
                    self.finish()
            else:
                if abs(gate_info["offset_h" if horizontal else "offset_v"]) > threshold:
                    self.finish()