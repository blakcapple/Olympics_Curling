import math 
import numpy as np



class Physical_Agent:

    gamma = 0.98
    delta_t = 0.1
    mass = 1 

    def __init__(self):
        self.theta = 90 
        self.pose = [300, 150]
        self.v =  [0, 0]
        self.acc = [0, 0]

    def reset(self):

        self.theta = 90 
        self.pose = [300, 150]
        self.v =  [0, 0]
        self.acc = [0, 0]

    def _action_to_accel(self, action):
        """
        Convert action(force) to acceleration
        """
        self.theta = self.theta + action[1]
        force = action[0] / self.mass
        accel_x = force * math.cos(self.theta / 180 * math.pi)
        accel_y = force * math.sin(self.theta / 180 * math.pi)
        accel = [accel_x, accel_y]
        self.acc = accel

    def _update_physics(self):
        x,y = self.pose
        vx, vy = self.v
        accel_x, accel_y = self.acc
        x_new = x + vx * self.delta_t  # update position with t
        y_new = y + vy * self.delta_t
        vx_new = self.gamma * vx + accel_x * self.delta_t  # update v with acceleration
        vy_new = self.gamma * vy + accel_y * self.delta_t
        self.v = [vx_new, vy_new]
        self.pose = [x_new, y_new]

    def step(self, action):
        # 更新加速度，更新角度
        self._action_to_accel(action)
        # 更新智能体速度和位置
        self._update_physics()


def calculate_pos(obs, colour, ego_pos=[300, 186]):
    scalar_y = 10.5
    scalar_x = 10.5
    """calculate pos given color"""
    color_pos = [] # relative pos in the obs
    color_group = [[] for _ in range (4)]
    match_number = 5 if colour == 'purple' else 1
    color_point = np.argwhere(obs == match_number)

    if color_point.shape[0] > 0:
        # gruop point
        for x, y in color_point:
            for i, point_group in enumerate(color_group):
                if len(point_group) == 0:
                    color_group[i].append([x,y])
                    break
                elif np.abs((x+y)/2 - np.mean(point_group)) < 2:
                    color_group[i].append([x,y])
                    break 

    for i, point_group in enumerate(color_group):
        if len(point_group) > 0:
            x_mean = np.mean([point_group[i][0] for i in range(len(point_group))])
            y_mean = np.mean([point_group[i][1] for i in range(len(point_group))])
            color_pos.append([x_mean, y_mean])
    relative_pos = color_pos
    real_pos = []
    for pos in relative_pos:
        pos_x = 450 - pos[1]*scalar_x
        pos_y = ego_pos[1]+30*scalar_y - pos[0]*scalar_y
        real_pos.append([pos_x, pos_y])
    return real_pos

def write_to_file(file, goal, reward):
    with open(file, 'a') as file_object:
        file_object.write(f'goal:{goal}; ')
        file_object.write(f'reward:{reward} \n')


def get_reward(pos, center):

    if pos[1] < 352:
        reward = -10
        return reward
    distance = math.sqrt((pos[0]-center[0])**2 + (pos[1]-center[1])**2)
    reward = 1 / (distance + 1) * 1000# distance reward 

    return reward