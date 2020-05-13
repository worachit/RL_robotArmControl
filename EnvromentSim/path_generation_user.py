from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.baxter import BaxterLeft, BaxterRight
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape

SCENE_FILE = join(dirname(abspath(__file__)), 'scene_baxter_pick_and_pass.ttt')
pr = PyRep()

pr.launch(SCENE_FILE, headless=False)
pr.start()

baxter_left = BaxterLeft()
baxter_right = BaxterRight()
baxter_gripper_left = BaxterGripper(0)
baxter_gripper_right = BaxterGripper(1)

cup = Shape('Cup')
waypoints = [Dummy('waypoint%d' % i) for i in range(7)]

print('Planning path for left arm to cup ...')
path = baxter_left.get_path(position=waypoints[0].get_position(), quaternion=waypoints[0].get_quaternion())
path.visualize()  # Let's see what the path looks like
print('Executing plan ...')
done = False
while not done:
    done = path.step()
    pr.step()
path.clear_visualization()

print('Done ...')
input('Press enter to finish ...')
pr.stop()
pr.shutdown()
