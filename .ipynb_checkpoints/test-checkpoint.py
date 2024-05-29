# print("hello world")

import matplotlib.pyplot as plt
plt.plot([1,2,3], [10, 20, 30])
# plt.savefig("/root/2024 NN Curriculum Design/output/")
plt.show()

# import matplotlib
# matplotlib.use('TkAgg')

# #---------------------------------------------------------------------------------------------
# import matplotlib
# print(matplotlib.__version__)

#---------------------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt

# plt.axis([0, 100, 0, 1])  #绘制横坐标0~100，纵坐标0~1
# plt.ion()  #开启交互模式，使图形可以动态更新。

# xs = [0, 0]
# ys = [1, 1]

# for i in range(100):
#     y = np.random.random()
#     xs[0] = xs[1]
#     ys[0] = ys[1]
#     xs[1] = i
#     ys[1] = y
#     plt.plot(xs, ys)
#     plt.pause(0.1)


#---------------------------------------------------------------------------------------------
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# # print(torch.__version__)

# x_train = np.array([[2.3],[4.4],[3.7],[6.1],[7.3],[2.1],[5.6],[7.7],[7.7],[4.1],[6.7],[6.1],[7.5],[2.1],[7.2],[5.6],[5.7],[7.7],[3.1]],dtype=np.float32)
# #生成矩阵数据
# y_train = np.array([[2.7],[4.76],[4.1],[7.1],[7.6],[3.5],[5.4],[7.6],[7.9],[5.3],[7.3],[7.5],[7.5],[3.2],[7.7],[6.4],[6.6],[7.9],[4.9]],dtype=np.float32)

# plt.figure()  #plt.figure()用于创建一个新的图形窗口或激活一个已存在的图形窗口。它通常用于设置图形的整体属性和外观，例如大小、分辨率、背景颜色

# plt.scatter(x_train,y_train)  #plt.scatter()函数用于画散点图，参数是上面定义的两个数组
# plt.xlabel('x_axis')  #x轴名称
# plt.ylabel('y_axis')  #y轴名称
# plt.show()  #显示图片

#---------------------------------------------------------------------------------------------
# # 使用range
# for i in range(1,5,2):
#     print(i)  # 输出 0, 1, 2, 3, 4


#---------------------------------------------------------------------------------------------
# # 使用arange
# import numpy as np
# arr = np.arange(5)
# print(arr)  # 输出 array([0, 1, 2, 3, 4])
