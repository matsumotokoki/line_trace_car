import numpy as np
from matplotlib import pyplot as plt
from line_trace_car import line_trace_car
import field
from collections import deque
import csv

BLACK     = '\033[30m'
RED       = '\033[31m'
GREEN     = '\033[32m'
YELLOW    = '\033[33m'
BLUE      = '\033[34m'
PURPLE    = '\033[35m'
CYAN      = '\033[36m'
WHITE     = '\033[37m'
END       = '\033[0m'
BOLD      = '\033[1m'
UNDERLINE = '\033[4m'
def Color(num):
    if num == 0:
        return BLACK
    elif num == 1:
        return WHITE
    elif num == 2:
        return BOLD
    elif num == 3:
        return CYAN
    elif num == 4:
        return PURPLE
    elif num == 5:
        return BLUE
    elif num == 6:
        return GREEN
    elif num == 7:
        return YELLOW
    elif num == 8:
        return RED
    else :
        return UNDERLINE

def get_state(car,s_state,v_state):
    state = [i != 0 for i in s_state]
    state.extend([car.mtrL,car.mtrR])
    return sum(np.array(state) * (np.array([1,2,4,8,16,(line_trace_car.car_order[1]-line_trace_car.car_order[0]+1)*16])))

def decide_action(next_state,episode,q_table):
    first_probability = 0.75
    epsilon = first_probability * (1/(episode+1))
    if epsilon <= np.random.uniform(0,1):
        next_action = np.argmax(q_table[next_state])
    else:
        prob = sum(q_table[next_state]+100)
        w = (q_table[next_state]+100) / prob
        next_action = np.random.choice(range(9) ,p=w) 
    return next_action

def update_Qtable(q_table,state,action,reward,next_state):
    gamma = 0.8
    alpha = 0.4
    next_max_q = max(q_table[next_state])
    q_table[state,action] = (1 - alpha) * q_table[state,action] + alpha * (reward + gamma * next_max_q)
    return q_table

#TODO adjustment
def update_reward(v_state,s_state,done,level):
        reward = 0

#0626
        if sum(v_state)/2 <= 1:
            reward = -1 
        elif s_state[0] or s_state[3]:
            reward += -0.1
        elif not done:
            reward += level 
        elif done and level == 8:
            reward += 500
        else:
            reward = -10

        return reward

def run():
    legend_flag = False
    max_episode = 50000
    step_by_episode = 1500
    goal_ave = 7
    review_num = 10
    reward_of_episode = 0
    reward_ave = np.full(review_num,0)
    learining_is_done = False
    q_table = np.random.uniform(low=-1,high=1,size=(2**4 * (line_trace_car.car_order[1]-line_trace_car.car_order[0]+1) ** 2 , line_trace_car.action_space))
    transition_data = [] 
    x_axis = []
    np.random.seed(0)
    for episode in range(max_episode):
        level = 0
        car=line_trace_car(field.n_line)
        s_state = car.get_sense()
        v_state = None 
        state  = get_state(car,s_state,v_state) 
        action = np.argmax(q_table[state])
        reward_of_episode = 0
        reward = 0

        for i in range(step_by_episode):
            v_state = car.step(action)[0]
            s_state = car.get_sense()
            done = car.checck_done(car.get_sense())
            reward = update_reward(v_state,s_state,done,level) 
            reward_of_episode += reward
            level = max(max(s_state), level)
            
            next_state = get_state(car,s_state,v_state)
            q_table = update_Qtable(q_table,state,action,reward,next_state)
            action = decide_action(next_state,episode,q_table)
            state = next_state
            # if learining_is_done or episode % 3000 == 0:
            if learining_is_done:
                car.plot_state()
                field.plot_normal_field_line(plt)
                car.path_plot(plt)
                if not legend_flag:
                    plt.legend()
                    legend_flag = True
                # plt.axes().set_aspect('equal')
                # plt.draw()
                # plt.pause(0.00001)
            
            if done:
                legend_flag = False
                reward_ave = np.hstack((reward_ave[1:],level))
                print(Color(level),end="")
                print("episode %5d, reward %6d, step %5d, x:%5d, y:%5d, level %d, reward_ave %f" %(episode+1,reward_of_episode,i+1,car.pos_x,car.pos_y,level,reward_ave.mean()))
                print(END,end="")
                transition_data.append(reward_ave.mean())
                x_axis.append(episode)
                #if learining_is_done == 1 or episode % 3000 == 0:
                if learining_is_done == 1 :
                    plt.axes().set_aspect('equal')
                    plt.show()
                    # plt.close()
                break 


        if (reward_ave.mean() >= goal_ave) or  episode+1 == max_episode:
            print("Episode %d train agent fin!" %(episode+1))
            with open('./csv/file.csv', 'wt') as f:
                writer = csv.writer(f)
                writer.writerows(q_table)
            print("saved")
            with open('./csv/transition_data.csv', 'wt') as f:
                writer = csv.writer(f)
                data = np.c_[x_axis,transition_data]
                writer.writerows(data)
            learining_is_done = 1
run()
