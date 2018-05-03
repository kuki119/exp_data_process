#### 将原始的大程序拆分######
##在导数据阶段完成采样  处理数据阶段直接计算！！
## 尝试在主程序中使用多线程计算 节省读取数据的时间
## 在计算特征量时使用多进程计算 发挥多核优势
## 开始使用git进行程序管理

import csv
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading as td 
import seaborn as sns
from queue import Queue

from otherFunc_multicore import *
np.seterr(divide='ignore',invalid='ignore')

class DataProcess(object):

    def __init__(self,file_path,doc_name):
        # 传入文件的 所处路径 和 文件名

        self.idx = findIdx(doc_name)
        self.path = os.path.join(file_path,doc_name)
        self.ptc_tim,self.scn_tim,self.time_ls,self.unit = self.inputData() #实现数据的分块 分类
        self.main_scn_area = self.calMainScnArea()
        self.main_scn_length = self.getMainScnAreaLen()
        # print('the main screen area is: <%f'%self.main_scn_area)
        # print('the main screen length is: %f'%self.main_scn_length)
        self.scn_len = self.getScreenLen()
        self.x_bod = self.getXBod() ##入料柱位置
        self.scr_eff = self.calScrEff(dim=0.9) ; print('efficiency:',self.scr_eff)

        self.end_time = self.getEndTime()  # 获取整体筛分结束时刻
        self.main_scn_end_tim = self.getMainScnEndTime() # 获取主筛区域筛分结束时刻

        # print('the screening end time index:',self.end_time)

    def __str__(self):

        return 'This is the data of experiment: ' + self.path

    def findIdx(self,doc_name):
        ## 传入实验数据文件名 返回当前实验标号
        idx1 = doc_name.find('_')
        idx2 = doc_name.find('.')
        exp_lab = doc_name[idx1+1:idx2]
        return int(exp_lab)

    def findStr(self,string,col1):
        #传入目标字符串 和 DataFrame型数据的一列  返回指定字符串的行标号
        #标记出指定字符串的序列
        # string = 'TIME:'
        str_lb = np.empty(col1.count(string),dtype='int') #存放指定字符串所在行号
        num = 0
        for i in np.arange(len(str_lb)):
            str_lb[i] = col1.index(string,num)
            num = int(str_lb[i]) + 1 #标号必须是整数 所以加 int()
            # print(num,i)
        return str_lb

    def sampling(self,freq,array,interval):
        # 输入数据  和 数据循环的频率
        # interval = 0.001
        period = 1.0/freq

        points = int(period // interval)
        num_periods = int(len(array) // points)

        last_points_lb = num_periods * points ## 要提取的最后一个数据点的标号
        sampled_points_num = 8 * num_periods ## 一个周期里提取8个点，总共提取sampled_points_num个点
        samples_lb = np.linspace(0,last_points_lb,sampled_points_num,dtype='int')

        df_array = pd.DataFrame(array,columns=['val'])
        array_sampled = df_array.iloc[samples_lb]
        return array_sampled.val

    def inputData(self):
        #导入数据  数据是 .csv格式
        data = []
        # path_now = os.path.join(path,file)
        with open(self.path) as csvfile:
            reader = csv.reader(csvfile)
            for i in reader:
                data.append(i)

        col1 = list(range(len(data))) #取出数据表中的第一列
        for i,j in enumerate(data):
            if len(j) > 1:
                col1[i] = j[0]

        lb_time = self.findStr('TIME:',col1)  #时间的行号 时间下的8行为数据
        lb_unit = self.findStr('UNITS:',col1) #单位的行号
        
        ##进行数据采样， 一个振动周期内均匀采8个点！
        # sampled_time = self.sampling(freq,lb_time,interval)

        time_ls = []
        for i in lb_time[1:]:
            time_ls.append(data[i][1])

        unit = []
        for i in lb_unit:
            unit.append(data[i][1])
        print('the units of data:',unit)

        #把各个时刻下的颗粒数据与筛网数据分开存放
        length = lb_time[1:] #可以指定从第几个时刻开始导入  #可以指定从第几个时刻开始导入
        ptc = list(range(len(length)))
        scn = list(range(len(length)))
        for i,j in enumerate(length):
            for num in range(1,10):
                if col1[j+num].find('mass') != -1: #不等于-1 意思是 当今标号是mass
                    mass = np.array(data[j+num][1:],dtype='float32')

                elif col1[j+num].find('ptc_id') != -1:
                    pid = np.array(data[j+num][1:],dtype=np.float32)

                elif col1[j+num].find('ptc_x') != -1:
                    x = np.array(data[j+num][1:],dtype='float32')

                elif col1[j+num].find('ptc_y') != -1:
                    y = np.array(data[j+num][1:],dtype='float32')

                elif col1[j+num].find('ptc_z') != -1:
                    z = np.array(data[j+num][1:],dtype='float32')

                elif col1[j+num].find('scn_xmax') != -1:
                    xmax = np.array(data[j+num][1:],dtype='float32')

                elif col1[j+num].find('scn_xmin') != -1:
                    xmin = np.array(data[j+num][1:],dtype='float32')

                elif col1[j+num].find('scn_zmax') != -1:
                    zmax = np.array(data[j+num][1:],dtype='float32')

                elif col1[j+num].find('scn_zmin') != -1:
                    zmin = np.array(data[j+num][1:],dtype='float32')
                else:
                    print('something wrong!!!')

            ptc[i] = pd.DataFrame(np.vstack([pid,x,y,z,mass]).T,columns=['pid','x','y','z','mass'])
            scn[i] = np.vstack([xmin,zmax,xmax,zmin]).T

        scn_tim = np.empty([len(scn),len(scn[0][0])])
        for i in range(len(scn)):
            scn_tim[i] = scn[i][0]

        time_ls = np.array(time_ls,dtype='float32')
        print('Import data has done!!')
        # return data,col1
        return ptc,scn_tim,time_ls,unit

    def getScreenLen(self):
        #根据筛网端部两点坐标计算筛长实际值
        p1 = self.scn_tim[0][0:2]
        p2 = self.scn_tim[0][2:]
        scn_len = np.sqrt(sum((p2-p1)**2))
        return scn_len

    def getMainScnAreaLen(self):
        # 根据主筛区域的 横坐标值 和 筛网的两端点坐标 推算 主筛区域 筛长
        kd = calLine(self.scn_tim[0])
        p_left = self.scn_tim[0][0:2]  # 筛网的左端点

        f = lambda x: kd[0] * x + kd[1]
        z = f(self.main_scn_area)
        
        p_main = np.array([self.main_scn_area,z])
        length = np.sqrt(sum((p_left-p_main)**2))
        return length

    def getXBod(self):
        # 计算料柱的 右侧边缘 x坐标
        ti = 10 #指定用于计算料柱的时刻
        ptc = self.ptc_tim[ti]

        z_lb = ptc.z.iloc[-len(ptc.z)//50]
        bool_lb = ptc.z > z_lb
        x_bod = ptc.x.loc[bool_lb].max() + 1 #把该值认为是 料柱的位置
        print('x_bod:',x_bod)
        # self.ptcPlot(ti,x_bod)
        return x_bod

    def ptcPlot(self,ti,x=0):
        #传入要画散点图的 颗粒位置数据 时刻值 和 想画的竖直线 x坐标
        ptc = self.ptc_tim[ti]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ptc[:,1],ptc[:,3],'.',c='k',alpha=0.3)
        ax.axvline(x)
        # ax.plot(ptc_newup[:,0],ptc_newup[:,2],'.',c='r',alpha=0.7)
        plt.show() #坐标转换后的展示图
        # import time
        time.sleep(5)

    def calLine(self,points):
        #传入要求直线 的 两个点坐标 返回该直线的参数
        #计算二维直线参数
        #传入两个点坐标的数组[x_min,z_max,x_max,z_min] 返回直线的两个参数
        #使用 np.linalg.solve 计算
        p = points
        # print(p[0])
        a = np.array([[p[0],1],[p[2],1]])
        b = np.array([p[1],p[3]])
        params = np.linalg.solve(a,b)
        return params

    def getEndTime(self):
        #计算筛分结束时刻  1、以筛面上颗粒数量减少率或最大颗粒数量的10%来定；2、以筛分效率稳定时刻来定
        tim_len = len(self.time_ls)

        num_up_ptc = np.array([self.getUpperPtc(ti) for ti in range(tim_len)]) ##计算每一个时刻对应的筛上颗粒数目
        middle = len(num_up_ptc) // 2
        rest = sum(num_up_ptc[middle:] > (num_up_ptc.max()*0.1) ) #大于 最大颗粒数量的10%的 最后一个时刻
        time_lb = middle + rest
        print('there still has {} on the deck when time={}'.format(num_up_ptc[time_lb],self.time_ls[time_lb]))

        return time_lb

    def getMainScnEndTime(self):
        #计算筛分结束时刻  1、以筛面上颗粒数量减少率或最大颗粒数量的10%来定；2、以筛分效率稳定时刻来定
        tim_len = len(self.time_ls)
        x_max = self.main_scn_area

        num_up_ptc = np.array([self.getUpperPtc(ti,x_max) for ti in range(tim_len)]) ##计算每一个时刻对应的筛上颗粒数目
        middle = len(num_up_ptc) // 2
        rest = sum(num_up_ptc[middle:] > (num_up_ptc.max()*0.1) ) #大于 最大颗粒数量的10%的 最后一个时刻
        time_lb = middle + rest
        print('there still has {} on the main screen area when time={}'.format(num_up_ptc[time_lb],self.time_ls[time_lb]))

        return time_lb

    def calZmin(self):
        """ 传入要计算的时刻值即可  计算过筛颗粒的位置
        根据所有颗粒中  x坐标 小于0 的颗粒的 z坐标最小值 """
        ptc = self.ptc_tim[-1]
        bool_x_less0 = ptc.x < 0
        zmin = ptc.z[bool_x_less0].min()
        return zmin

    def calStdMass(self,diam=0.9):
        #传入目标 直径  返回 该直径所对应的质量
        # diam = 0.9 #指定分类粒径
        dns = 2678  #密度2678千克每立方米
        diam = diam/1000 #将 毫米 单位化成 米
        mass = ((np.pi/6)*dns*diam**3)*1000 #把质量的kg单位变成g单位 分离粒径颗粒质量
        return mass

    def getUnderPtc(self,ti,x_max):
        #输入 想要计算的时刻值  返回该时刻下的  筛下颗粒信息
        z_min = self.calZmin() #剔除掉已经筛过的颗粒， 可能不同的实验设置会出错！！！
        # x_max = self.scn_tim[ti][2]
        ptc = self.ptc_tim[ti]
        kb = self.calLine(self.scn_tim[ti])
        bool_und = (ptc.x < x_max) & (ptc.z > z_min) & ((ptc.z - ptc.x * kb[0]-kb[1])<0)
        ptc_und = ptc.loc[bool_und]

        # ptc_und.plot.scatter('x','z')
        # plt.show()
        return ptc_und

    def getUpperPtc(self, ti, x_max=None):
        #输入 想要计算的时刻值  返回该时刻下的  筛上颗粒信息 

        ptc = self.ptc_tim[ti]
        kb = self.calLine(self.scn_tim[ti])
        
        if x_max == None:
            bool_up = ((ptc.z - ptc.x * kb[0]-kb[1]) > 0)
        else:
            bool_up = (ptc.x <= x_max) & ((ptc.z - ptc.x * kb[0]-kb[1]) > 0)
        
        ptc_up = ptc.loc[bool_up]

        # ptc_up.plot.scatter('x','z')
        # plt.show()
        return ptc_up.shape[0]

    def calScrEff(self,ti=-1,dim=0.9):  #默认情况下计算最后一刻时刻的筛分效率
        #计算筛分效率  传入要计算的 时刻值
        # dim = 0.9 #指定分离粒径
        # z_min = self.calZmin() #剔除掉已经筛过的颗粒， 可能不同的实验设置会出错！！！
        # x_max = self.scn_tim[ti][2]
        ptc = self.ptc_tim[ti]
        # kb = self.calLine(self.scn_tim[ti])
        # bool_und = (ptc.x < x_max) & (ptc.z > z_min) & ((ptc.z - ptc.x * kb[0]-kb[1])<0)
        # ptc_und = ptc.loc[bool_und]
        x_max = self.scn_tim[ti][2]
        ptc_und = self.getUnderPtc(ti,x_max)
        mass_std = self.calStdMass(dim) #传入指定直径返回相应的球形颗粒质量
        bool_all_sml = ptc.mass < mass_std
        bool_und_sml = ptc_und.mass < mass_std

        eff_sml = ptc_und.mass[ptc_und.mass < mass_std].sum() / ptc.mass[ptc.mass < mass_std].sum()
        eff_lag = ptc_und.mass[ptc_und.mass > mass_std].sum() / ptc.mass[ptc.mass > mass_std].sum()
        eff = eff_sml - eff_lag

        # plt.plot(ptc[bool_all_sml,0],ptc[bool_all_sml,1],'.',c='k',alpha=0.3) #画小颗粒分布
        # plt.plot(ptc[~bool_all_sml,0],ptc[~bool_all_sml,1],'.',c='g',alpha=0.1) #画大颗粒分布
        # plt.show() #验证图
        # return eff,eff_sml,eff_lag
        return eff

    def getConvertMatrix(self,points):
        #传入筛网的两个点
        #转换矩阵 新坐标系基底在旧坐标系里的投影坐标
        #转换矩阵 [cos(a),sin(a);-sin(a),cos(a)]  二维……
        p = points
        dp = np.array([abs(p[0]-p[2]),abs(p[1]-p[3])])
        modp = np.sqrt(sum(dp**2)) #求向量的模
        sinp = dp[1]/modp
        cosp = dp[0]/modp
        #构造 二维转换矩阵 和 四维转换矩阵
        con_mat2d = np.array([[cosp,-sinp],[sinp,cosp]]).T #二维转换矩阵
        # con_matrix = np.array([[cosp,0,-sinp],[0,1,0],[sinp,0,cosp]]).T #三维转换矩阵 绕y轴转p角度
        con_mat4d = np.array([[cosp,0,-sinp,0],[0,1,0,0],[sinp,0,cosp,0],[0,0,0,1]]).T #四维转换矩阵
        con_mat5d = np.array([[1,0,0,0,0],[0,cosp,0,-sinp,0],[0,0,1,0,0],[0,sinp,0,cosp,0],[0,0,0,0,1]]).T #四维转换矩阵
        return con_mat2d,con_mat4d,con_mat5d

    def calMainScnArea(self,ti=-1):
        ### 返回筛下 80% 的颗粒的x坐标
        kd = self.calLine(self.scn_tim[ti])
        ptc = self.ptc_tim[ti]
        bool_under = (ptc.x<self.scn_tim[ti][2]) & ((ptc.z-(kd[0]*ptc.x+kd[1]))<0)
        ptc_und = ptc.loc[bool_under]
        num_under_total = sum(bool_under)

        desc = ptc_und.x.describe(percentiles=[0.8])
        x_label = desc['80%']
        print('main screen arean:<',x_label)

        return x_label

    def getBedPtc(self,ti):
        # 主筛区域内 的 一部分颗粒
        #传入要计算料层颗粒的 时刻值！
        # ti=id_stab_tims[0]
        kd = self.calLine(self.scn_tim[ti])
        ptc = self.ptc_tim[ti]
        # bool_up = (ptc_xyz[:,0]>self.x_bod)&(ptc_xyz[:,0]<self.scn_tim[ti][2])&((ptc_xyz[:,2]-(kd[0]*ptc_xyz[:,0]+kd[1]))>0)
        
        x_label = self.main_scn_area  # x_label左侧认为是主要筛分区域
        x_bod = self.x_bod ##入料柱位置  以入料柱右侧的主要筛分区域为 判断料层高度的区域
        
        bool_up_part = (ptc.x > x_bod) & (ptc.x < x_label) & ((ptc.z -(kd[0]*ptc.x +kd[1]))>0) #去除料柱 0503
        bool_up_total = (ptc.x < x_label) & ((ptc.z -(kd[0]*ptc.x +kd[1]))>0) #去除料柱 0503
        ptc_up_part = ptc.loc[bool_up_part]
        ptc_up_total = ptc.loc[bool_up_total]

        #尝试使用坐标转换矩阵
        cm2,cm4,cm5 = self.getConvertMatrix(self.scn_tim[ti])
        ptc_newup_part = np.dot(ptc_up_part,cm5);print(ptc_newup_part.shape)
        ptc_newup_total = np.dot(ptc_up_total,cm5);print(ptc_newup_total.shape)
        ptc_newup_part = pd.DataFrame(ptc_newup_part,columns = ['pid','x','y','z','mass'])
        ptc_newup_total = pd.DataFrame(ptc_newup_total,columns = ['pid','x','y','z','mass'])

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(ptc_up[:,0],ptc_up[:,2],'.',c='k',alpha=0.3)
        # ax.plot(ptc_newup_part[:,0],ptc_newup_part[:,2],'.',c='r',alpha=0.7)
        # plt.show() #坐标转换后的展示图

        # import time
        # time.sleep(5)

        ##使用去除料柱的数据计算料层上下界面
        desc = ptc_newup_part.z.describe(percentiles=[0.05,0.85])
        bed_top = desc['85%']
        bed_bottom = desc['5%']
    
        ## 使用没有去除入料柱的数据计算新的料层数据
        bool_bed = (ptc_newup_total.z > bed_bottom)&(ptc_newup_total.z < bed_top)
        ptc_bed = ptc_newup_total[bool_bed]
        return ptc_bed, np.array([bed_bottom,bed_top])
        # return ptc_newup_part, ptc_newup_total

    def calDiam(self,mass):
        #传入颗粒质量 返回球形颗粒直径
        dns = 2678 #密度2678千克每立方米
        mass = mass/1000 #将 g 单位化成 kg
        c = 6/(dns*np.pi)
        diam = (mass*c)**(1/3)
        diam = diam*1000 #将米 转成 毫米
        return diam

path = 'E:\\data\\features_data'
exp = DataProcess(path,'exp_24.csv')

ptc_bed, bed_lb = exp.getBedPtc(100)


