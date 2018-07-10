import os

from exp_process_multicore import *
from otherFunc_multicore import *
np.seterr(divide='ignore',invalid='ignore')

def parallel(doc,q):

    path = 'E:\\data_1'
    exp = DataProcess(path,doc)
    # idx_ = exp.idx
    # print(exp)
    # ptc_,stra_,poro_,pene_ = calFeatures(exp)
    # ptc_ = calFeatures(exp,q)
    # idx_,ptc_,stra_,poro_,pene_ = calFeatures(exp)
    idx_,ptc_,pene_ = calFeatures(exp)

    # q.put(idx_,ptc_,stra_,poro_,pene_)
    q.put(idx_,ptc_)

def main():
    path_parent = 'F:\\diff_model_data\\'
    path_child = os.listdir(path_parent)

    # path3 = 'E:\\experiments_wang\\lang\\data_8'
    # path = 'L:\\shiyan-changshi\\1\\data_1'
    for lb, pa in enumerate(path_child):
        path = os.path.join(path_parent, pa)

        if lb > -1:
            model_lb = path[-2:]
            docs = os.listdir(path)
            # hz = [16,18,20,22,24,20,20,20,20]
            print('第%d个文件夹，下有%f个文件'%(lb,len(docs)))
            # q = Queue()
            dic = dict(idx=[],eff=[],unit_eff=[],scr_time=[],main_scn_ratio=[],ptc_num=[],
                touch_m=[],touch_v=[],stra_m=[],stra_v=[],poro_x=[],poro_z=[],pene=[])
            
            # ##使用多线程 
            # batch = 2 ##指定一次计算几个实验
            # # for j in range(int(len(docs))//batch):
            # for i in range(1): #先计算前4个文件 创建4个线程 
            #     t = td.Thread(target=parallel, args=(docs[i],q))
            #     t.start()
            # for _ in range(1):
            #     t.join()

            # for _ in range(1):
            #     res = q.get()
            #     # dic['idx'].append(res[0])
            #     # dic['ptc_num'].append(res[1])
            #     # dic['stra'].append(res[2])
            #     # dic['poro'].append(res[3])
            #     # dic['pene'].append(res[4])
            #     print(res,'>>>')

            ## 不使用多线程
            for d_,doc in enumerate(docs):
            # doc = docs[0]
                # if d_ == 4:

                exp = DataProcess(path,doc)
                print(exp)
                main_scn_ratio = exp.main_scn_length / exp.scn_len
                eff = exp.scr_eff
                scr_time = exp.end_time
                unit_eff = exp.scr_eff / scr_time
                # main_scr_time = exp.time_ls[exp.main_scn_end_tim]
                
                idx_,ptc_,touch_m,touch_v,stra_m,stra_v,poro_x_,poro_z_,pene_ = calFeatures(exp)
                # idx_,ptc_,touch_ = calFeatures(exp)
                
                dic['idx'].append(idx_)
                dic['main_scn_ratio'].append(main_scn_ratio)
                dic['scr_time'].append(scr_time)
                dic['ptc_num'].append(ptc_)
                dic['stra_m'].append(stra_m)
                dic['stra_v'].append(stra_v)
                dic['poro_x'].append(poro_x_)
                dic['poro_z'].append(poro_z_)
                dic['pene'].append(pene_)
                # dic['bed_h'].append(bed_h_)
                dic['touch_m'].append(touch_m)
                dic['touch_v'].append(touch_v)
                dic['eff'].append(eff)
                dic['unit_eff'].append(unit_eff)
                # dic['main_scr_time'].append(main_scr_time)

                # print(dic)

            df = pd.DataFrame(dic)
            df.to_excel('..\\features\\Features_0708_'+ model_lb +'.xlsx')

if __name__ == '__main__':
    main()