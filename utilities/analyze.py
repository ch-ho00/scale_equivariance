import pandas as pd

result = []
labels = ["dataset","batch_size", "model", "base", "io_scale", "n_scale","interaction","channel", "epoch", "init_rate", "decay", 'gamma',"step_size","save_dir",'scale_1', 'scale_1.1','scale_1.5','scale_2','scale_2.5','scale_3']
for file_ in ['dss_v1.txt','dss_v2.txt']: 
    with open(file_,'r',encoding='utf-8-sig') as f:
        l = " "
        while len(l) != 0:
            l = f.readline()
            if '--' in l or 'Bessel' in l or 'Test for' in l:
                continue
            elif 'dss' in l:
                l= l.split(' ')
                l[4] = l[4][1:2]
                try:
                    del l[5], l[-1]
                except:
                    continue
                add = []
                add += l
                enter = 0
            elif "Test Accuracy" in l:
                add.append(l[l.find(':')+1:l.find('/')].strip())
                enter += 1
            if enter == 6 or (('dss_v1' in file_) and (enter==4)):
                
                enter = 0
                result.append(add)
            
result = pd.DataFrame(result,columns=labels)
result.to_csv('./dss_mnist.csv')