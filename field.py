import numpy as np
from numpy import pi 
from matplotlib import pyplot as plt

n_line = []
t_line = []

n_line.append(lambda x: -10<=x[0]<=10 and -10<=x[1]<=1150)
n_line.append(lambda x: x[0]<=150 and 1150<=x[1] and 140<=np.linalg.norm(np.array(x)-np.array([150,1150]))<=160)
n_line.append(lambda x: 150<=x[0]<=300 and 1290<=x[1]<=1310)
n_line.append(lambda x: 300<=x[0]<=890 and 1290<=x[1]<=1310)
n_line.append(lambda x: 890<=x[0] and 1150<=x[1] and 140<=np.linalg.norm(np.array(x)-np.array([890,1150]))<=160)
n_line.append(lambda x: 1030<=x[0]<=1050 and 1000<=x[1]<=1150)
n_line.append(lambda x: 1030<=x[0]<=1050 and 900<=x[1]<=1000)
n_line.append(lambda x: 1030<=x[0]<=1050 and 800<=x[1]<=900)

nplot_x1 = [10]
nplot_y1 = [0]
nplot_x2 = [-10]
nplot_y2 = [0]

for th in (np.linspace(0,pi/2,45)):
    nplot_x1.append(150-140*np.cos(th))
    nplot_y1.append(1150+140*np.sin(th))
    nplot_x2.append(150-160*np.cos(th)) 
    nplot_y2.append(1150+160*np.sin(th)) 

for th in (np.linspace(0,pi/2,45)):
    nplot_x1.append(890+140*np.sin(th))
    nplot_y1.append(1150+140*np.cos(th))
    nplot_x2.append(890+160*np.sin(th)) 
    nplot_y2.append(1150+160*np.cos(th)) 

nplot_x1.append(1030)
nplot_y1.append(900)
nplot_x2.append(1050)
nplot_y2.append(900)

def plot_train_field_line(plt):
    plt.plot(tplot_x1,tplot_y1,color="black")
    plt.plot(tplot_x2,tplot_y2,color="black")
def plot_normal_field_line(plt):
    plt.plot(nplot_x1,nplot_y1,color="black")
    plt.plot(nplot_x2,nplot_y2,color="black")

plot_normal_field_line(plt)
plt.axes().set_aspect('equal')
plt.show()
