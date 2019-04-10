"""
Flight Simulator
Author: Yuto Terauchi
"""
from math import sin, cos, tan, sqrt
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 機体パラメータ設定
#--------------------------------#
# 有次元安定微係数
Xu = -0.01
Zu = 0.1
Mu = 0.001

Xa = 30
Za = -200
Ma = -4

Xq = 0.3
Zq = -5
Mq = -1

Yb = -45
Lb_= -2
Nb_= 1

Yp = 0.5
Lp_= -1
Np_= 0.1

Yr = 3
Lr_= 0.2
Nr_=-0.2
#--------------------------------#

g  = 9.8 # 重力加速度

# その他のパラメタ
W0 = 0.0
U0 = 100
theta0 = 0.05

# 縦のシステム
A_lat = np.matrix(
        [[Xu,    Xa,    -W0,        -g*cos(theta0)],
         [Zu/U0, Za/U0, (U0+Zq)/U0, -g*sin(theta0)/U0],
         [Mu,	   Ma,    Mq,	      0],
         [0,	   0,     1,          0]])

# 横・方向のシステム
A_lon = np.matrix(
        [[Yb,  (W0+Yp), -(U0-Yr),      g*cos(theta0), 0],
         [Lb_, Lp_,     Lr_,           0,             0],
         [Nb_, Np_,     Nr_,           0,             0],
         [0,   1,       tan(theta0),   0,             0],
         [0,   0,       1/cos(theta0), 0,             0]])

# 計算条件の設定
endurance = 100 # 飛行時間[sec]
dt        = 0.01 #1.0[sec]あたりの時間ステップ数
time = np.arange(0, endurance, dt)

# 初期値 x0_lat = [u,alpha,q,theta], x0_lon = [beta,p,r,phi,psi]
x0_lat = np.matrix([10.0, 0.1, 0.4, 0.2]).T # 縦の初期値
x0_lon = np.matrix([0.0, 0.6, 0.4, 0.2, 0.2]).T # 横・方向の初期値
x0_pos = np.matrix([0.0, 0.0, -1000]).T # 初期位置

def rotation_x(psi):
    r_x = np.matrix(
          [[cos(psi),  sin(psi), 0],
           [-sin(psi), cos(psi), 0],
           [0,         0,        1]])
    return r_x

def rotation_y(theta):
    r_y = np.matrix(
          [[cos(theta), 0, -sin(theta)],
           [0,          1, 0],
           [sin(theta), 0, cos(theta)]])
    return r_y

def rotation_z(phi):
    r_z = np.matrix(
          [[1, 0,         0],
           [0, cos(phi),  sin(phi)],
           [0, -sin(phi), cos(phi)]])
    return r_z

def system(x_lat, x_lon):
    dx_lat = A_lat * x_lat
    dx_lon = A_lon * x_lon
    velocity = (x_lat[0,0] + U0) * np.matrix([1.0, x_lon[0,0], x_lat[1,0]]).T
    dx_pos = rotation_x(-x_lon[4,0]) * rotation_y(-x_lat[3,0]) * rotation_z(-x_lon[3,0]) * velocity
    return dx_lat, dx_lon, dx_pos

def explicit_euler(x_lat, x_lon, x_pos):
    dx_lat, dx_lon, dx_pos = system(x_lat, x_lon)
    x_lat = x_lat + dx_lat * dt
    x_lon = x_lon + dx_lon * dt
    x_pos = x_pos + dx_pos * dt
    return x_lat, x_lon, x_pos

def RK4(x_lat, x_lon, x_pos):
    k1_lat, k1_lon, dx_pos = system(x_lat, x_lon)
    k2_lat, k2_lon, dx_pos = system(x_lat + dt/2 * k1_lat, x_lon + dt/2 * k1_lon)
    k3_lat, k3_lon, dx_pos = system(x_lat + dt/2 * k2_lat, x_lon + dt/2 * k2_lon)
    k4_lat, k4_lon, dx_pos = system(x_lat + dt * k3_lat, x_lon + dt * k3_lon)
    x_lat = x_lat + dt/6 * (k1_lat + 2*k2_lat + 2*k3_lat + k4_lat)
    x_lon = x_lon + dt/6 * (k1_lon + 2*k2_lon + 2*k3_lon + k4_lon)
    x_pos = x_pos + dx_pos * dt
    return x_lat, x_lon, x_pos


def main():
    x_lat, x_lon, x_pos = x0_lat, x0_lon, x0_pos
    position_history = [np.asarray(x_pos).reshape(-1)]
    for _ in time:
        # x_lat, x_lon, x_pos = explicit_euler(x_lat, x_lon, x_pos)
        x_lat, x_lon, x_pos = RK4(x_lat, x_lon, x_pos)
        # if x_pos[2,0]<0:
        #     distance = sqrt((x_pos[0,0] - position_history[0][0])**2 + (x_pos[1,0] - position_history[0][1])**2)
        #     print("{0:.2f}".format(distance))
        #     break
        position_history.append(np.asarray(x_pos).reshape(-1))
    position_history = np.array(position_history)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, np.max(position_history[:,2]))
    ax.plot(position_history[:,0], position_history[:,1], position_history[:,2])
    plt.show()

if __name__ == "__main__":
    main()