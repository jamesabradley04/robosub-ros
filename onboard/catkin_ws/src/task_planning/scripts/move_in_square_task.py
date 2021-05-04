from task import Task
from move_tasks import MoveToPoseLocalTask, MoveToPoseGlobalTask
from combination_tasks import ListTask
import task_utils


class MoveInSquareTask(Task):
    """Move the robot in a square at a certain depth."""

    def __init__(self, movement_distance, depth=0):  # maybe replace this with a dict or list
        """
        Parameters:
            move_distance(float): distance of each leg of the square path
            depth(float): depth to keep the movement at
        """
        super(MoveInSquareTask, self).__init__()

        self.step_one = MoveToPoseGlobalTask(movement_distance, 0, depth, 0, 0, 0)
        self.step_two = MoveToPoseGlobalTask(0, movement_distance, depth, 0, 0, 0)
        self.step_three = MoveToPoseGlobalTask(-movement_distance, 0, depth, 0, 0, 0)
        self.step_four = MoveToPoseGlobalTask(0, -movement_distance, depth, 0, 0, 0)
        self.list_task = ListTask([self.step_one, self.step_two, self.step_three, self.step_four])

    def _on_task_start(self):
        pass

    def _on_task_run(self):
        self.list_task.run()
        if(self.list_task.finished):
            self.finish()
