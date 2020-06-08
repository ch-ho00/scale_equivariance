from scipy.ndimage import zoom
import torch
import numpy as np
import matplotlib.pyplot as plt 
def scale(img, zoom_factor, **kwargs):
    '''
    Input
        img = input image (H,W,B,C) 
        zoom_factor = scale factor
    Output:
        scaled image = (H,W,B,C)
    ''' 
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out implying zero padding towards outside
    if zoom_factor < 1:
        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
    # Zooming in
    elif zoom_factor > 1:
        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        #  trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)

        if trim_top<0:
            out = img
        else:
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    # If zoom_factor == 1, just return the input array
    else:
        out = img
    assert out.shape == img.shape
    return out


def cal_dss_equiv_err(sc_em, reg_em,s,level=1,base=1.2,n_scale=4):
    '''
    Calculate the DSS equivariance error between one scale level of scaled embedding and regular embedding through scale zoom
    Input: 
        sc_em = scaled images' embedding ,torch.tensor(H,W,B,C)
        reg_em = original images' embedding ,torch.tensor(H,W,B,C)
        s = scale between original image and transformed image
        level = scale level to be considered
        base, io_scale = hyperparemeter of model
    Output:
        equivariance error (i.e. sum || Ls(f(x)) - f(Ls(x)) ||_2 )
    '''
    sum_ = 0
    k = int(max(round(np.log(s)/np.log(base)),0))
    if isinstance(reg_em,list):
        k = min(k+level,n_scale-1)
        sc_em_pred = reg_em[k].permute(2,3,0,1).detach().cpu().numpy()
        sc_em_true = sc_em[level].permute(2,3,0,1).detach().cpu().numpy()
        for i in range(sc_em_pred.shape[2]):
            for j in range(sc_em_pred.shape[3]):
               sum_ += np.linalg.norm(sc_em_pred[:,:,i,j]- sc_em_true[:,:,i,j],2)/  np.linalg.norm(sc_em_true[:,:,i,j],2)
    else:
        sc_em = sc_em.cpu().numpy()
        reg_em = reg_em.cpu().numpy()
        k = min(k+level,sc_em.shape[2]-1)
        sc_em_pred = reg_em[:,:,k,:,:]
        for i in range(sc_em_pred.shape[2]):
            for j in range(sc_em_pred.shape[3]):
                sum_ += np.linalg.norm(sc_em_pred[:,:,i,j]- sc_em[:,:,level,i,j],2)/  np.linalg.norm(sc_em[:,:,level,i,j],2)
    return sum_


  
def cal_equiv_err(sc_em, reg_em,s):
    '''
    Calculate the equivariance error between scaled embedding and regular embedding through scale zoom
    Input: 
        sc_em = scaled images' embedding ,torch.Tensor(H,W,B,C)
        sc_em = scaled images' embedding ,torch.Tensor(H,W,B,C)
        s = scale between original image and transformed image
    Output:
        equivariance error (i.e. sum || Ls(f(x)) - f(Ls(x)) ||_2 )
    '''
    sc_em_pred = scale(reg_em, s)
    sc_em_pred = torch.tensor(sc_em_pred)
    sc_em_pred = sc_em_pred.cpu().numpy()
    sc_em = sc_em.cpu().numpy()
    sum_ = 0
    for i in range(sc_em_pred.shape[2]):
        for j in range(sc_em_pred.shape[3]):
            sum_ += np.linalg.norm(sc_em_pred[:,:,i,j]- sc_em[:,:,i,j],2)/  np.linalg.norm(sc_em[:,:,i,j])
    return sum_

def plot_graph(equiv_list, acc_list,fname):
    fig1= plt.figure(1)
    plt.title('%s Accuracy across scale transformation'%(fname))
    plt.plot([a[0] for a in acc_list], [a[1] for a in acc_list])
    plt.xlabel("scale")
    plt.ylabel("Accuracy")
    fig1.savefig('../results/metric_graph/%s_acc.PNG'%(fname))
    fig2= plt.figure(2)
    plt.title('%s Equivariance Error across scale transformation'%(fname))
    plt.plot([a[0] for a in equiv_list], [a[1] for a in equiv_list])
    plt.xlabel("Scale")
    plt.ylabel("Equivariance error")
    fig2.savefig('../results/metric_graph/%s_equiv.PNG'%(fname))
    plt.show()
    plt.close()
