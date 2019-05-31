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
    first_probability = 0.5
    epsilon = first_probability * (1/(episode+1))
    if epsilon <= np.random.uniform(0,1):
        next_action = np.argmax(q_table[next_state])
    else:
        prob = sum(q_table[next_state]+1)
        w = (q_table[next_state]+1) / prob
        next_action = np.random.choice(range(9) ,p=w) 
    return next_action

def update_Qtable(q_table,state,action,reward,next_state):
    gamma = 0.5
    alpha = 0.4
    next_max_q = max(q_table[next_state])
    q_table[state,action] = (1 - alpha) * q_table[state,action] + alpha * (reward + gamma * next_max_q)
    return q_table

def run():
    max_episode = 1000000
    step_by_episode = 10000
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

        for i in range(step_by_episode):
            v_state = car.step(action)[0]
            s_state = car.get_sense()
            done = car.checck_done(car.get_sense())
            if sum(v_state) == 0:
                reward = -0.75
            elif not done:
                reward = 0.05
            else:
                reward = done

            reward_of_episode += reward
            level = max(max(s_state), level)
            
            next_state = get_state(car,s_state,v_state)
            q_table = update_Qtable(q_table,state,action,reward,next_state)
            action = decide_action(next_state,episode,q_table)
            state = next_state
            if learining_is_done or episode % 1000 == 0:
                car.plot_state()
                field.plot_normal_field_line(plt)
                car.path_plot(plt)
                plt.draw()
                plt.pause(0.001)
            
            if done:
                reward_ave = np.hstack((reward_ave[1:],level))
                print("episode %5d, reward %5d, step %5d, x:%5d, y:%5d, level %d" %(episode+1,reward_of_episode,i+1,car.pos_x,car.pos_y,level))
                if learining_is_done == 1 or episode % 1000 == 0:
                    # field.plot_normal_field_line(plt)
                    # car.path_plot(plt)
                    # plt.show()
                    plt.clf()
                    plt.close()
                break


        if (reward_ave.mean() >= goal_ave):
            print("Episode %d train agent successfuly!" % episode)
            learining_is_done = 1
run()
