import numpy as np
from matplotlib import pyplot as plt
import line_trace_car
import field
from collections import deque

def get_state(car,act_v):
    return car.get_sense() 

def get_action(car,state):
    order_left = car.car_order[0]
    order_right = car.car_order[0]

    if state[3] != 0:
        order_left = 1
    elif state[2] == 0:
        order_left = -1
    if state[0]  != 0:
        order_right = 1
    elif state[1] == 0:
        order_right = -1
    
    if (car.mtrL + order_left) <=0 and (car.mtrR +order_right)<=0:
        order_left = 1
        order_right =1
    if order_left == 0 and  order_right == 0:
        order_left = -1 if car.mtrL > 5 else(1 if car.mtrL < 5 else 0)
        order_left = -1 if car.mtrR > 5 else(1 if car.mtrR < 5 else 0)
    return (order_left+1)*3 + (order_right+1)


def main():
    num_episodes = 100
    max_steps = 50000
    total_steps  = 0
    goal_level = 8 
    average_episodes = 100
    goal_ave = 0
    level_deque  = deque(maxlen = average_episodes)
    islearned = True
    for episode in range(num_episodes):
        car=line_trace_car.line_trace_car(field.n_line)
        state = get_state(car,car.step()[0])
        stage = 1
        
        for i in range(max_steps):
            action = get_action(car,state)
            v,action = car.step(action)
            next_state = get_state(car,v)
            sense = np.array(car.get_sense())
            done = car.checck_done(sense) if i+1 < max_steps  else -1
            reward = done

            if islearned:
                car.plot_state()
                field.plot_normal_field_line(plt)
                car.path_plot(plt)
                plt.axes().set_aspect('equal')
                plt.draw()
                plt.pause(0.00001)

            if done:
                level_deque.append(stage)
                print("%5d (%5d, %5d) %4d %2d %.2f %5d" %(episode+1,car.pos_x,car.pos_y,i+1,stage,sum(level_deque)/len(level_deque),done))
                goal_ave += i+1
                break
            
            stage = min(sense[sense.nonzero()])
            total_steps += 1
            state = next_state

            if islearned:
                car.plot_state()

        if islearned:
            if done > 0 or episode+1 == num_episodes:
                print("ave: " + str(goal_ave/episode))
                car.path_plot(plt)
                field.plot_normal_field_line(plt)
                plt.axes().set_aspect('equal')
                plt.legend(loc="lower right")
                plt.show()

        if  sum(level_deque)/len(level_deque) >= goal_level or episode + 1 == num_episodes-1:
            print("complete learning!!")
            islearned = True

if __name__ == "__main__":
    main()
