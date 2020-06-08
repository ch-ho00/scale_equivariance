import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def interaction(data,dataset,model="dss_v1"): 
    for int_ in data['interaction'].unique():
        d = [int_,pd.DataFrame.mean(data[(data.model== model) ^ (data.interaction ==int_) ^ (data.scale_3 > 5000)][['scale_1.1','scale_1.5','scale_2','scale_3']])]
        d = np.array([[a,b] for a,b in zip([1.1,1.5,2,3],d[1])])
        plt.figure(1)
        plt.plot(d[:,0],d[:,1]/10000,label='Interaction %d'%(int_))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('Effect of interaction for %s'%(model))
    plt.ylabel('Accuracy')
    plt.xlabel('Scale')
    plt.savefig('../results/dss_param_%s/%s_interaction.png'%(dataset,model))
    
    plt.close()

def n_scale(data,dataset,model="dss_v1"): 
    for n in data['n_scale'].unique():
        d = [n,pd.DataFrame.mean(data[(data.model== model) ^ (data.n_scale ==n) ^ (data.scale_3 > 5000)][['scale_1.1','scale_1.5','scale_2','scale_3']])]
        d = np.array([[a,b] for a,b in zip([1.1,1.5,2,3],d[1])])
        plt.figure(1)
        plt.plot(d[:,0],d[:,1]/10000,label='n_scale %d'%(n))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('Effect of n_scale for %s'%(model))
    plt.ylabel('Accuracy')
    plt.xlabel('Scale')
    plt.savefig('../results/dss_param_%s/%s_n_scale.png'%(dataset,model))
    
    plt.close()

def base(data,dataset,model="dss_v1"): 
    for n in data['base'].unique():
        d = [n,pd.DataFrame.mean(data[(data.model== model) ^ (data.base ==n) ^ (data.scale_3 > 5000)][['scale_1.1','scale_1.5','scale_2','scale_3']])]
        d = np.array([[a,b] for a,b in zip([1.1,1.5,2,3],d[1])])
        plt.figure(1)
        plt.plot(d[:,0],d[:,1]/10000,label='base %.3f'%(n))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('Effect of base for %s'%(model))
    plt.ylabel('Accuracy')
    plt.xlabel('Scale')
    plt.savefig('../results/dss_param_%s/%s_base.png'%(dataset,model))
    
    plt.close()

def io_scale(data,dataset,model="dss_v1"): 
    for n in data['io_scale'].unique():
        d = [n,pd.DataFrame.mean(data[(data.model== model) ^ (data.io_scale ==n) ^ (data.scale_3 > 5000)][['scale_1.1','scale_1.5','scale_2','scale_3']])]
        d = np.array([[a,b] for a,b in zip([1.1,1.5,2,3],d[1])])
        plt.figure(1)
        plt.plot(d[:,0],d[:,1]/10000,label='io_scale %d'%(n))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('Effect of io_scale for %s'%(model))
    plt.ylabel('Accuracy')
    plt.xlabel('Scale')
    
    plt.savefig('../results/dss_param_%s/%s_io_scale.png'%(dataset,model)) 
    plt.close()

def channel(data,dataset,model="dss_v1"): 
    for n in data['channel'].unique():
        d = [n,pd.DataFrame.mean(data[(data.model== model) ^ (data.channel ==n) ^ (data.scale_3 > 5000)][['scale_1.1','scale_1.5','scale_2','scale_3']])]
        d = np.array([[a,b] for a,b in zip([1.1,1.5,2,3],d[1])])
        plt.figure(1)
        plt.plot(d[:,0],d[:,1]/10000,label='channel %d'%(n))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('Effect of channel for %s'%(model))
    plt.ylabel('Accuracy')
    plt.xlabel('Scale')
    plt.savefig('../results/dss_param_%s/%s_channel.png'%(dataset,model))
    
    plt.close()

def init_rate(data,dataset,model="dss_v1"): 
    for n in data['init_rate'].unique():
        d = [n,pd.DataFrame.mean(data[(data.model== model) ^ (data.init_rate ==n) ^ (data.scale_3 > 5000)][['scale_1.1','scale_1.5','scale_2','scale_3']])]
        d = np.array([[a,b] for a,b in zip([1.1,1.5,2,3],d[1])])
        plt.figure(1)
        plt.plot(d[:,0],d[:,1]/10000,label='init_rate %.3f'%(n))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.title('Effect of init_rate for %s'%(model))
    plt.ylabel('Accuracy')
    plt.xlabel('Scale')
    
    plt.savefig('../results/dss_param_%s/%s_init_rate.png'%(dataset,model))
    plt.close()

data = pd.read_csv('dss_mnist.csv')
interaction(data,dataset,model)
n_scale(data,dataset,model)
base(data,dataset,model)
io_scale(data,dataset,model)
channel(data,dataset,model)
init_rate(data,dataset,model)


