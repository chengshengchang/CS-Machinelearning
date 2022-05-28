import numpy as np 
import sklearn.preprocessing as sp 
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

#from mpl_toolkits.mplot3d import axes3d 3d 繪圖工具

#線性回歸x,y 求 loss_wfuntion 最小 , 最佳回歸線
#x0 , x1 , loss , 求 min (loss)


# x data

xs = np.array([0.5,0.6,0.8,1.1,1.4])

# y data

ys = np.array([5.0,5.5,6.0,6.8,7.0])


#設定 w0 , w1 區間 決定 最小 loss 我們最佳線性函數 y= ax + b 
n = 500
w0_grid, w1_grid  = np.meshgrid(np.linspace(-10, 10,n),np.linspace(-10, 10,n))



#求總樣本誤差
loss = 0
for x ,y in zip(xs,ys):
    loss += (w0_grid + w1_grid *x -y )**2 /2
    
    
#畫圖

mp.figure('3d model of Loss_funtion',facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel =('w0')
ax3d.set_ylabel =('w1')
ax3d.set_zlabel =('loss')
ax3d.plot_surface(w0_grid,w1_grid,loss,cstride=30,rstride=30,cmap='jet')


mp.show(ax3d)

    
#找到最佳 ax+b 可以隨便找一點w0,w1 做梯度下降 得到模型最佳解


    
    