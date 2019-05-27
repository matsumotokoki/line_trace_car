import numpy as np
from numpy import pi 

class line_trace_car(object):
    sensors= np.array([[150,22.5],[150,7.5],[150,-7.5],[150,-22.5]])
    tread = 200
    car_order = [0,10]
    init_pos = np.array([[-10,10],[-10,10],[80*pi/180,100*pi/180]])
    action_space = 9 #3*3

    def __init__(self,lines):
        self.lines = lines
        while True:
            self.pos_x =  np.random.uniform(self.init_pos[0,0],self.init_pos[0,1])
            self.pos_y =  np.random.uniform(self.init_pos[1,0],self.init_pos[1,1]) - 150
            self.pos_a =  np.random.uniform(self.init_pos[2,0],self.init_pos[2,1])
            if any(self.get_sense()):
                break
        self.mtrL = self.car_order[0]
        self.mtrR = self.car_order[0]
        self.car_pos = [[],[]]
        self.sensor_pos = []
        for _ in range(len(self.sensors)):
            self.sensor_pos.append([[],[]])
    
    def get_sense(self): 
        sense = []
        angs = np.array([[np.cos(self.pos_a),-np.sin(self.pos_a)],[np.sin(self.pos_a),np.cos(self.pos_a)]])
        for i in range(len(self.sensors)):
            sensor_val = 0
            sensor_p = np.array([sum(self.sensors[i] * angs[0]) + self.pos_x,sum(self.sensors[i]  * angs[1]) +  self.pos_y])
            for j ,line in enumerate(self.lines):
                if line(sensor_p):
                    sensor_val = j+1
                    break
            sense.append(sensor_val)
        return sense

    def step(self,action=4):
        mL = np.clip(self.mtrL + action//3 -1, self.car_order[0],self.car_order[1])
        mR = np.clip(self.mtrR + action%3 -1, self.car_order[0],self.car_order[1])
        act_mL =np.random.uniform(0.45,0.55) * (self.mtrL + mL)
        act_mR =np.random.uniform(0.45,0.55) * (self.mtrR + mR)
        mid_angle =  self.pos_a + (act_mR-act_mL) / (4 * self.tread) # kinnji
        self.pos_x += np.cos(mid_angle) * (act_mL+act_mR)/2
        self.pos_y += np.sin(mid_angle) * (act_mL+act_mR)/2
        self.pos_a += (act_mR-act_mL)/(2 * self.tread)
        real = (mL - self.mtrL + 1) * 3 + (mR - self.mtrR + 1)
        self.mtrL = mL
        self.mtrR = mR
        return ([act_mL,act_mR],real)

    def checck_done(self,sense):
        sense = np.array(sense)
        done = 0
        if not any(sense):
            done = -1
        elif min(sense[sense.nonzero()]) == len(self.lines):
            done = 1
        return done
        
    def plot_state(self):
        self.car_pos[0].append(self.pos_x)
        self.car_pos[1].append(self.pos_y)
        angs = np.array([[np.cos(self.pos_a),-np.sin(self.pos_a)],[np.sin(self.pos_a),np.cos(self.pos_a)]])
        for i in range(len(self.sensors)):
            sensor_p = np.array([sum(self.sensors[i] * angs[0]) + self.pos_x,sum(self.sensors[i]  * angs[1]) +  self.pos_y])
            self.sensor_pos[i][0].append(sensor_p[0])
            self.sensor_pos[i][1].append(sensor_p[1])

    def path_plot(self,plt):
        plt.plot(self.car_pos[0],self.car_pos[1],label="car_pos")
        for i in range(len(self.sensors)):
            plt.plot(self.sensor_pos[i][0],self.sensor_pos[i][1],label="sensor"+str(i+1))
            
