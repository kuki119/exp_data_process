import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures 
import multiprocessing as mp
## 尝试使用多线程计算  ThreadPoolExecutor

def findIdx(doc_name):
    ## 传入实验数据文件名 返回当前实验标号
    idx1 = doc_name.find('_')
    idx2 = doc_name.find('.')
    exp_lab = doc_name[idx1+1:idx2]
    return int(exp_lab)

def calDiam(mass):
    #传入颗粒质量 返回球形颗粒直径
    dns = 2678 #密度2678千克每立方米
    mass = mass/1000 #将 g 单位化成 kg
    c = 6/(dns*np.pi)
    diam = (mass*c)**(1/3)
    diam = diam*1000 #将米 转成 毫米
    return diam

def calMass(diam):
    #传入颗粒质量 返回球形颗粒直径
    dns = 2678 #密度2678千克每立方米
    diam = diam/1000 #将 mm 单位化成 m
    c = (dns*np.pi) / 6
    mass = c * diam ** 3
    mass = mass*1000 #将米 转成 毫米
    return mass

def calLine(points):
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

def pickPoints(array):
    ## 传入pd.series数组 返回相对平滑阶段的 数据编号
    a1 = array[:-1]
    a2 = array[1:]
    a3 = a1 - a2 ## 相邻两点纵坐标之差 即斜率大小

    slop_max = a3.mean() + 0.8*a3.std() # 设定为斜率的最大值
    slop_min = a3.mean() - 0.8*a3.std() # 设定为斜率的最小值

    tag_max = (a3<slop_max).argmin() # 想要的最后一个数据
    tag_min = (a3<slop_min).argmin() # 想要的第一个数据
    return tag_min,tag_max+1

def calPeriodValue(array, period=8):
    period_num = (len(array)-1) // period 
    values = [array[i*period:(i+1)*period].mean() for i in range(period_num)]
    return np.array(values)

def calPickValue(ptc_array,target_array):
    ## 传入颗粒数量序列 和 目标数列，根据颗粒数量挑出平稳阶段 计算目标序列的均值
    pik = pickPoints(ptc_array)
    return ptc_array[pik[0]:pik[1]].mean(), target_array[pik[0]:pik[1]].mean()

def calStratification(ptc_bed,z_label):
    #传入要计算分层系数的 料层颗粒信息 和 料层厚度
    # print('stratification++++++'+str(len(ptc_bed)))
    # std_diam = 0.9
    # std_mass = calMass(std_diam)

    # ##尝试使用在料层中的小颗粒距离顶部的平均距离 与 料层厚度之比
    # sml_ptc = ptc_bed[ptc_bed.mass < std_mass] ##取出料层中的小颗粒
    # dist_top = z_label[1] - sml_ptc.z.mean()
    # deta_z = z_label[1] - z_label[0]
    # stra = dist_top / deta_z

    if ptc_bed.shape[0] < 2:
        r = np.nan
        return r

    ptc_bed['diam'] = calDiam(ptc_bed.mass)
    r = ptc_bed.corr().diam.z  # 去除 颗粒空间位置 z 坐标 与 颗粒直径尺寸的 相关系数

    # z = ptc_bed.z #颗粒z轴坐标
    # x = calDiam(ptc_bed.mass) #颗粒粒径
    # data = np.vstack([z,x]).T
    # df = pd.DataFrame(data)
    # r = df.corr().iloc[0,1] #给出z坐标与颗粒粒径的相关关系
    # #计算 x y 之间的相关性 与 斜率 推测分层情况
    # ###看一下料层里 沿着z轴 颗粒粒径 的分布情况
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x,y,'.')
    # ax.set(title=str(r),xlabel='particle daimeter',ylabel='Z',)
    # plt.savefig(str(ti)+'stratification.png')

    # #尝试将颗粒分为 大于分离粒径颗粒 和 小分离粒径颗粒 分别计算相关系数 然后取两者之比
    # ptc_sml = ptc_bed.loc[ptc_bed.diam < std_diam]
    # ptc_lag = ptc_bed.loc[ptc_bed.diam >= std_diam]
    # #
    # # r_sml = ptc_sml.corr().diam.z
    # # r_lag = ptc_lag.corr().diam.z
    # #
    # # r_new = r_sml/r_lag

    # ptc_sml['z_diam'] = ptc_sml.z / ptc_sml.diam # 小颗粒距离筛面距离远近
    # ptc_lag['z_diam'] = ptc_lag.z / ptc_lag.diam

    # stra_sml = ptc_sml.z_diam.mean()
    # stra_lag = ptc_lag.z_diam.mean()

    # r = stra_lag / stra_sml #希望该值越大越好

    # ### 尝试把料层中颗粒直径进行 归一化 处理，之后算统一的 相关系数
    # ptc_bed['diam_norm'] = (ptc_bed.diam - ptc_bed.diam.mean()) / ptc_bed.diam.std()
    #
    # r_new_norm = ptc_bed.corr().diam.z

    # return np.array([r_old,r_new,r_new_norm])
    return r
    # return stra

def calAllTimeStra(exp_obj):
    # 传入 实验数据  和  该实验所使用的 振动频率
    time = exp_obj.time_ls
    len_time = len(time)
    stra = np.zeros([len_time,1])
    ptc_bed_num = np.zeros([len_time,1])

    for ti in range(len_time):
        bed,z_labels = exp_obj.getBedPtc(ti)
        print('the number of particles in bed:', bed.shape[0])

        ptc_bed_num[ti] = bed.shape[0]  # 记录料层中 颗粒数量随时间的变化
        stra[ti] = calStratification(bed,z_labels)  # 记录每个时刻下的 z坐标与粒径 的相关系数

    # df_all = pd.DataFrame(np.hstack([r,ptc_bed_num]),columns=['R_z_diam','ptc_bed_num'])

    label = ptc_bed_num.nonzero() ## 相关系数数组中 的 非零项  包括 np.nan
    ptc_bed_num = ptc_bed_num[label]
    stra = stra[label]

    # ptc_bed_num_period,l = getPeriod(hz,ptc_bed_num,interval) 
    # stra_period = getPeriod(hz,r,interval)
    stra_period = calPeriodValue(stra)
    ptc_bed_num_period = calPeriodValue(ptc_bed_num)
    # print(ptc_bed_num_period.shape,'\n',r_period.shape)

    ptc_,stra_ = calPickValue(ptc_bed_num_period,stra_period)

    # df_period = pd.DataFrame(np.hstack([stra_period.reshape(-1,1),ptc_bed_num_period.reshape(-1,1)]),
    #     columns=['stra_period','ptc_num_period'])  ## 必须保证 1 列数据
    
    # df = pd.DataFrame(np.hstack([stra.reshape(-1,1),ptc_bed_num.reshape(-1,1)]),
    #     columns=['stras','ptc_num'])  ## 必须保证 1 列数据
    return ptc_,stra_
    # return df_period
    # return df_all,r,ptc_bed_num

def func(params):

    exp_obj = params[0]
    ti = params[1]
    bed,z_labels = exp_obj.getBedPtc(ti)
    # print('the number of particles in bed:', bed.shape[0])
    bed_h = z_labels[1] - z_labels[0]

    ptc_bed_num = bed.shape[0]  # 记录料层中 颗粒数量随时间的变化
    # stra = calStratification(bed,z_labels)  # 记录每个时刻下的 z坐标与粒径 的相关系数
    # poro_x,poro_z = Porosity(bed,bed_h,exp_obj.main_scn_length)  # 记录每个时刻下的 料层松散度

    ptc_und_num = exp_obj.getUnderPtc(ti).shape[0] ##计算筛下颗粒数量变化，观察稳筛阶段颗粒占比
    return ti,ptc_bed_num,ptc_und_num
    ### 因为多个时刻的数据同时计算，为了后续分辨哪个时刻数据，所以这里返回时刻值
    # return ti,ptc_bed_num,stra,poro_x,poro_z,bed_h

def calFeatures(exp_obj):
    ## 尝试使用并行计算 同时计算分层和松散
    idx_ = exp_obj.idx
    time = exp_obj.time_ls
    len_time = len(time)
    # stra = np.zeros([len_time,1])
    # poros = np.zeros([len_time,1]) # 记录各个时刻下的 松散
    # poros = np.zeros([len_time,1]) # 记录各个时刻下的 松散
    # ptc_bed_num = np.zeros([len_time,1])

    pool = mp.Pool(8)
    params = [[exp_obj,ti] for ti in range(len_time)]  ##map传入的参数必须是可迭代的，所以把exp_obj与ti组合成可迭代形式
    res = pool.map(func,params)
    print('multicore is done!!')
    pool.close()  ##关闭进程池！！
    pool.join()

    # print(res,'\n',len(res))
    np_res = np.array(res) ##第一列为时刻值，后续列为所计算特征
    pd_res = pd.DataFrame(np_res,columns=['ti','ptc_bed_num','stra','poros_x','poros_z','bed_h']) ## 各个列名称为返回的数据
    pd_res = pd_res.sort_values(by='ti')
    # ptc_bed_num = pd_res.ptc_bed_num
    print(pd_res.tail())

    label = pd_res.ptc_bed_num.nonzero()[0] ## 相关系数数组中 的 非零项  包括 np.nan 由于已经不是np.array数据类型了 更改
    ptc_bed_num = pd_res.ptc_bed_num[label]
    stra = pd_res.stra[label]
    poros_x = pd_res.poros_x[label]
    poros_z = pd_res.poros_z[label]
    bed_h = pd_res.bed_h[label]

    bed_h_period = calPeriodValue(bed_h)
    poros_x_period = calPeriodValue(poros_x)
    poros_z_period = calPeriodValue(poros_z)
    stra_period = calPeriodValue(stra)
    pene_period = Penetration(exp_obj)
    ptc_bed_num_period = calPeriodValue(ptc_bed_num)

    ptc_,pene_ = calPickValue(ptc_bed_num_period,pene_period)
    _,stra_ = calPickValue(ptc_bed_num_period,stra_period)
    _,poro_x_ = calPickValue(ptc_bed_num_period,poros_x_period)
    _,poro_z_ = calPickValue(ptc_bed_num_period,poros_z_period)
    _,bed_h_ = calPickValue(ptc_bed_num_period,bed_h_period)

    # return idx_,ptc_,pene_
    return idx_,ptc_,stra_,poro_x_,poro_z_,pene_,bed_h_

def periodLastLabel(array, period=8):
    period_num = (len(array)-1) // period  ## 缩短目标序列，避免超出index
    labels = [(i+1)*period for i in range(period_num)]
    return np.array(labels)

def Penetration(exp_obj):
    ## 后一个时刻  主筛区域 筛下新增颗粒  与 前一个时刻筛上颗粒 之比
    ## 还需要导入 后一时刻的 筛网位置坐标
    ## 还是得用 前后两个周期 进行比较  前后两个时刻间隔太小 会出现后一个时刻筛下颗粒数目减小的现象
    ## 传入目标实验的频率值  计算每个周期的最后一个时刻  比对这些时刻的颗粒

    time = exp_obj.time_ls
    labels = periodLastLabel(time)  # 以平动周期对时间序列进行分割 返回最后一个时刻的标号
    # print(labels)
    # print(time.shape)
    # print(len(exp_obj.ptc_tim))
    x_max = exp_obj.main_scn_area
    penetration_ratio = np.zeros([len(labels),1])

    i_ = 0
    for ti_1, ti_2 in zip(labels[0:-1],labels[1:]):  # 用前后两个周期的 最后一个时刻计算
        ptc_tim1_up = exp_obj.getUpperPtc(ti_1)  # 仅仅主筛区域以上颗粒
        ptc_tim1_und = exp_obj.getUnderPtc(ti_1,x_max)  # 仅仅主筛区域以下颗粒
        ptc_tim2_und = exp_obj.getUnderPtc(ti_2,x_max)

        ptc_id_tim1_und = set(ptc_tim1_und.pid)
        ptc_id_tim2_und = set(ptc_tim2_und.pid)
        diff_tim2_tim1 = findDiff(ptc_id_tim1_und,ptc_id_tim2_und)  # 筛下新增的颗粒
        # print('length of difference between tim1 and tim2:',len(diff_tim2_tim1))

        ptc_num_tim1_up = ptc_tim1_up.shape[0]
        # print(ptc_num_tim1_up)
        i_ += 1
        if ptc_num_tim1_up > 0: 
            # num_new_ptc.append(len(diff_tim2_tim1))
            # num_tim1_up.append(ptc_num_tim1_up)
            penetration_ratio[i_] = (len(diff_tim2_tim1) / ptc_num_tim1_up)
        # else:
        #     penetration_ratio.append(np.nan)
        #     break  # 如果筛上颗粒数量很少时 则跳出循环

        # diff_tim2_sub_tim1 = ptc_id_tim2_und.difference(ptc_id_tim1_und);print('len of tim2 minus tim1:',len(diff_tim2_sub_tim1))

    # df = {'penetration_ratio': penetration_ratio,'num_new_ptc': num_new_ptc, 'num_tim1_up': num_tim1_up}
    # df = pd.DataFrame(df)
    return penetration_ratio

def findDiff(array1,array2):
    ## 输入两个数组np.array  返回 两者交集以外的 数字
    ## 使用集合 set(a).symmetric_difference(set(b)) 取出两集合的不同项目
    ## 或者使用  set(a).difference(set(b))  取出集合 a 中 b 所没有的
    a = set(array1)
    b = set(array2)
    diff = a.symmetric_difference(b)
    return diff

def getPeriod(freq,array,interval):
    # 输入数据  和 数据循环的频率
    # interval = 0.001
    period = 1.0/freq

    points = int(period // interval)
    num_periods = int(len(array) // points)

    mean_period = np.array([array[i*points : (i+1)*points].mean() for i in range(num_periods)])
    period_last_label = np.array([(i+1)*points for i in range(num_periods)]) - 1 # 每个周期的最后一个时刻的标号
    
    # for i in range(int(num_periods)):
    #     print('*****')
    #     plt.plot(array[i*points : (i+1)*points])
    #     print('*****')
    # plt.show()
    return mean_period, period_last_label

def Porosity(ptc_bed,delta_z,main_scn_length):
    ## 主筛区域 料层内 所有颗粒投影到 x轴上 两两之间 距离的均值 与 主筛区域长度 之比

    coord_x = ptc_bed.x  ## 第一次计算 计算x方向上的松散程度 显示出负相关 
    coord_z = ptc_bed.z  ## 第二次尝试 计算z方向上的松散程度  使用z轴方向上的距离均值 与 料层厚度之比 
    
    ### 使用 均方根的 方法 会出现负值
    # mat_total = np.dot(coord.values.reshape(-1,1),coord.values.reshape(1,-1)) # (x1)^2 - 2*x1*x2 + (x2)^2 的所有项
    # mat_dist = np.empty_like(mat_total)
    # row = mat_dist.shape[0]
    # col = mat_dist.shape[1]
    # for i in range(row):
    #     for j in range(col):

    #         temp = mat_total[i,i] - 2*mat_total[i,j] + mat_total[j,j]  ## 从预先计算好的矩阵中取出 进行距离计算
    #         # print(coord.iloc[i],coord.iloc[j])
    #         # print(mat_total[i,i], mat_total[i,j], mat_total[j,j])
    #         if temp < 0:
    #             print('^^^^'*10,temp)
    #             print('xi:{0},xj:{1}'.format(coord[i],coord[j]))
    #             print('mat[i,i]:{0},mat[i,j]:{1},mat[j,j]:{2}'.format(mat_total[i,i],mat_total[i,j],mat_total[j,j]))
    #             return coord,mat_total
    #         mat_dist[i,j] = np.sqrt(temp)

    ## 使用 横坐标取差值 之后 再取绝对值
    mat_ones = np.ones([len(coord_x),len(coord_x)])
    mat_col_x = coord_x.values.reshape(-1,1) * mat_ones
    mat_row_x = coord_x.values.reshape(1,-1) * mat_ones
    mat_sub_x = mat_col_x - mat_row_x  # 所有颗粒的横坐标 两两取差值
    mat_dist_x = np.abs(mat_sub_x) # 对坐标差值 取绝对值
    
    mat_col_z = coord_z.values.reshape(-1,1) * mat_ones
    mat_row_z = coord_z.values.reshape(1,-1) * mat_ones
    mat_sub_z = mat_col_z - mat_row_z  # 所有颗粒的横坐标 两两取差值
    mat_dist_z = np.abs(mat_sub_z) # 对坐标差值 取绝对值

    # print(mat_dist[0:5,0:5])
    num_elems = ((mat_dist_x.shape[0]*mat_dist_x.shape[1] - mat_dist_x.shape[0])) ## 距离矩阵的上三角\下三角的元素个数
    dist_mean_x = mat_dist_x.sum() / num_elems
    dist_mean_z = mat_dist_z.sum() / num_elems
    
    poro_x = dist_mean_x / main_scn_length  # 使用x轴方向上 颗粒之间的距离均值 与 主筛长 之比 得出：松散程度与综合筛分效率负相关
    poro_z = dist_mean_z / delta_z  ## 尝试使用 z轴方向上，颗粒之间的距离均值 与 料层厚度之比！
    # print(dist_mean,main_scn_length)
    return poro_x,poro_z 

def calAllTimePoro(exp_obj):
    # 传入 实验数据  和  该实验所使用的 振动频率
    time = exp_obj.time_ls
    len_time = len(time)
    poros = np.zeros([len_time,1]) # 记录各个时刻下的 透筛概率
    ptc_bed_num = np.zeros([len_time,1])

    for ti in range(len_time):
        bed,z_labels = exp_obj.getBedPtc(ti)
        print('the number of particles in bed:', bed.shape[0])
        delta_z = z_labels[1] - z_labels[0]

        if bed.shape[0] < 2:
            break

        ptc_bed_num[ti] = bed.shape[0]  # 记录料层中 颗粒数量随时间的变化
        poros[ti] = Porosity(bed,delta_z,exp_obj.main_scn_length)  # 记录每个时刻下的 料层松散度
        print('porosity of the ptc bed:',poros[ti])

    # df_all = pd.DataFrame(np.hstack([poros,ptc_bed_num]),columns=['R_z_diam','ptc_bed_num'])

    label = ptc_bed_num.nonzero() ## 相关系数数组中 的 非零项  包括 np.nan
    ptc_bed_num = ptc_bed_num[label]
    poros = poros[label]

    # ptc_bed_num_period,l = getPeriod(hz,ptc_bed_num,interval) 
    # poros_period,l = getPeriod(hz,poros,interval)
    # print(ptc_bed_num_period.shape,'\n',r_period.shape)

    poros_period = calPeriodValue(poros)
    ptc_bed_num_period = calPeriodValue(ptc_bed_num)
    df_period = pd.DataFrame(np.hstack([poros_period.reshape(-1,1),ptc_bed_num_period.reshape(-1,1)]),
        columns=['porosity_period','ptc_num_period'])  ## 必须保证 1 列数据

    # df = pd.DataFrame(np.hstack([poros.reshape(-1,1),ptc_bed_num.reshape(-1,1)]),
    #     columns=['porosity','ptc_num'])  ## 必须保证 1 列数据

    return df_period

def getIntervalPoints(array, num_points):
    ## 传入一个数组 以及 想要采几个点
    ## 实现从 目标数组中 进行 均匀的采样

    end_label = len(array) - 1
    new_label = np.linspace(0,end_label,num_points,endpoint=False,dtype=np.int) # 采样值的标号

    array_new = array[[new_label]] ## 对目标数组 进行 均匀采样

    return array_new 

def statSmallPtcInBed(ptc_bed,std_diam=1.0):
    ## 统计料层中小颗粒数量 随时间变化情况
    ## 传入料层颗粒信息 和 筛网尺寸
    ptc_bed['diam'] = calDiam(ptc_bed.mass)
    bool_small = ptc_bed.diam < std_diam
    num_small_ptc = sum(bool_small)
    return num_small_ptc

def statAllTimeSmallPtcInBed(exp_obj,hz,interval=0.001):
    # 传入 实验数据  和  该实验所使用的 振动频率
    time = exp_obj.time_ls
    len_time = len(time)
    num_small_ptc = np.zeros([len_time,1]) # 记录各个时刻下的 料层中小颗粒数量
    ptc_bed_num = np.zeros([len_time,1])

    for ti in range(len_time):
        bed,delta_z = exp_obj.getBedPtc(ti)
        # print('the number of particles in bed:', bed.shape[0])

        if bed.shape[0] < 2:
            break

        ptc_bed_num[ti] = bed.shape[0]  # 记录料层中 颗粒数量随时间的变化
        num_small_ptc[ti] = statSmallPtcInBed(bed) 
        print('num of small ptc:',num_small_ptc[ti])

    # df_all = pd.DataFrame(np.hstack([num_small_ptc,ptc_bed_num]),columns=['R_z_diam','ptc_bed_num'])

    label = ptc_bed_num.nonzero() ## 相关系数数组中 的 非零项  包括 np.nan
    ptc_bed_num = ptc_bed_num[label]
    num_small_ptc = num_small_ptc[label]

    ptc_bed_num_period,l = getPeriod(hz,ptc_bed_num,interval) 
    num_small_ptc_period,l = getPeriod(hz,num_small_ptc,interval)
    # print(ptc_bed_num_period.shape,'\n',r_period.shape)

    df_period = pd.DataFrame(np.hstack([num_small_ptc_period.reshape(-1,1),ptc_bed_num_period.reshape(-1,1)]),
        columns=['num_small_ptc_period','num_ptc_period'])  ## 必须保证 1 列数据

    return df_period

def main():
    df54 = pd.read_csv('./stra_5_4.csv')
    df54 = df54.drop(0)

    df14 = pd.read_csv('./stra_1_4.csv')
    df14 = df14.drop(0)

    stra14,l = getPeriod(16,df14.old)
    stra54,l = getPeriod(24,df54.old)

    plt.plot(stra14,label='exp_14')
    plt.plot(stra54,label='exp_54')
    plt.legend(loc = 'best')
    plt.show()

if __name__ == '__main__':
    main()
