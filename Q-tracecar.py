import numpy as np
from matplotlib import pyplot as plt
from line_trace_car import line_trace_car
import field
from collections import deque
import csv

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
    gamma = 0.70
    alpha = 0.5
    next_max_q = max(q_table[next_state])
    q_table[state,action] = (1 - alpha) * q_table[state,action] + alpha * (reward + gamma * next_max_q)
    return q_table

#TODO adjustment
def update_reward(v_state,s_state,done,level):
        reward = 0
        # if sum(v_state) <= 0:
        #     reward += -10 
        # elif sum(v_state)/2 < 2:
        #     reward += -1
        if s_state[0] or s_state[3]:
            reward = 0
        # if not done:
        #     reward += 0.5 
        # elif done and level == 8:
        #     reward += 500
        # else:
        #     reward = -100
        # return reward
        elif not done:
            reward = round(sum(v_state))   
        elif done and level > 5:
            reward = 100 * level
        else:
            reward = -99
        return reward


def run():
    legend_flag = False
    max_episode = 100000
    step_by_episode = 5000
    goal_ave = 7
    review_num = 10
    reward_of_episode = 0
    reward_ave = np.full(review_num,0)
    learining_is_done = False
    q_table = np.random.uniform(low=-1,high=1,size=(2**4 * (line_trace_car.car_order[1]-line_trace_car.car_order[0]+1) ** 2 , line_trace_car.action_space))
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
                plt.draw()
                plt.pause(0.00001)
            
            if done:
                legend_flag = False
                reward_ave = np.hstack((reward_ave[1:],level))
                print("episode %5d, reward %6d, step %5d, x:%5d, y:%5d, level %d" %(episode+1,reward_of_episode,i+1,car.pos_x,car.pos_y,level))
                #if learining_is_done == 1 or episode % 3000 == 0:
                if learining_is_done == 1 :
                    plt.close()
                break


        if (reward_ave.mean() >= goal_ave):
            print("Episode %d train agent successfuly!" % episode)
            with open('./csv/file.csv', 'wt') as f:
                writer = csv.writer(f)
                writer.writerows(q_table)
            learining_is_done = 1
run()
