from task import Task
from move_tasks import MoveToPoseLocalTask
import task_utils

class MoveInSquareTask(Task):
    """Move the robot in a square at a certain depth."""

    def __init__(self, movement_distance, depth=0): # maybe replace this with a dict or list
        """
        Parameters:
            move_distance(float): distance of each leg of the square path
            depth(float): depth to keep the movement at
        """
        super(MoveInSquareTask, self).__init__()

        self.step_one = MoveToPoseLocalTask(movement_distance, 0, depth, 0, 0, 0)
        self.step_two = MoveToPoseLocalTask(0, movement_distance, depth, 0, 0, 0)
        self.step_three = MoveToPoseLocalTask(-movement_distance, 0, depth, 0, 0, 0)
        self.step_four = MoveToPoseLocalTask(0, -movement_distance, depth, 0, 0, 0)

    
    def _on_task_start(self):
        pass
    
    def _on_task_run(self):
        self.step_one.run()
        self.step_two.run()
        self.step_three.run()
        self.step_four.run()

        self.finish()