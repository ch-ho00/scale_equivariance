# Scale Equivariance
Analysis of equivariance under scale transformation in CNN.
For definitions and details relating to the models and ideas being incorporated, please refer to the appendix.

Furthermore [document](https://github.com/chanhopark00/scale_equivariance/blob/master/Scale_Equivariance_Chan_Ho_park.pdf) is made to compare different models that achieve scale equivariance please refer if one wishes to see more ideas on scale equivariance. 

To run the code, refer to build_classification.py and equivariance_test.py.

## 1. Aim

To propose a convolution where scale equivariance is preserved to an extent through scale convolution while in class local transformation is captured by deformable convolution.

### 1.1. Scale Invariance

Scale invariance is generally achieved through the decreasing size of the feature map through max pooling and data augmentation. An example of this would be models like ResNet reduces the dimension of feature map through where max pooling and increasing stride from 1 to 2 after for each begining of layers. This can be intuitively seen as the orientation of features of a particular object being preserved regardless of scale of the instance. Similar ideas can be found in Feature Pyramid Network (FPN) where the decreasing size of feature maps leads to scale invariance.
 
### 1.2. Deformable Convolution

One major downside of regular convolution is that the receptive field of the kernel is in a fixed grid form. This implies that the features being extracted are in a fixed position. However in most cases, objects can come in any variations such as rotated, tilted or scaled.  

Methods such as Spatial Tranformer Network (STN) attempts to predict the transformation applied to the original object. The transformation predicted is an affine transformation. However this method still has an limitation of not being able to capture the non-linear transformation applied. The structure of STN convolution modules and the transformation applied looks as such: 
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/stn.png" width="300" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/stn2.png" width="300">

Hence in order to deal with the in-class variations, we make use of deformable convolution. Similar to the structure of STN, deformable convolution has an additional parallel convolution branch that predicts the offset of the grid. This allows the receptive fields of the convolution to be more effective. The structure and exemplar receptive field looks as such: 
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/dcn.png" width="300">
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/dcn2.png" width="300">

Therefore for the improvement of performance, we attempt incorporation of deformable convolution into scale convolution of deep scale space in the later part.

### 1.3. Deep Scale Space

A model that achieves scale equivariance, Deep Scale Space (DSS), involves a scale dimension. The scale dimension is generated by applying a gaussian kernel on the input image. The effect of gaussian kernel can be seen as a way of bandlimitting the image in a way such that the evident edges and features remain while the rest blurs. The authors refer this process as "lifting". Hence after lifting the dimension of image/feature map is as [#images, #channels, #scale dimension, width, height] 

An demonstration of applying the kernel is as such:

<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/dss.PNG" width="200">

Then a dilated convolution is applied in a way such that depending on the scale level that we are considering. The higher the scale level, the more dilation onto the kernel. With the bandlimitting through gaussian kernel, we can see this as a way to ensure that features can be extracted regardless of the scale of the instance. 

Note that the kernel applied in each k=0 , k=1 and k=2 are all tied weights.

<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/dss2.PNG" >

Lastly before the task-specific convolution or fully connected layer, the model applies a scale pooling which is an average pooling opertaion across the scale dimension. After scale pooling the dimensionality of the feature map becomes [#images, #channels, width, height]

## 2. Suggested Convolution

In order to improve the performance of DSS model, we have attempted to employ two new methods. To explain more about each the models,

### 2.1. DSS v2

Instead of using Gaussian Kernel to downsample the input image, we reduce the dimension of the input space through bilinear interpolation. So in this case the "lifting" operation can be seen as bilinear interpolation. Before the fully connected layer, the lifted feature maps of different dimensions go through max pooling in order to match with dimension of the smallest feature map. 

### 2.2. Deformable DSS

Keeping the convolution structure of the original DSS, we add a parallel convolution branch for predicting the offset of the kernel.  

## 3. Equivariance in existing models

Before proceeding to the new methodologies, we first set out baseline through the experiments on existing models. In order to check in which extent scale equivariance is achieved, we make use of the formula,

<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/equivariance_error.PNG" >

Here, Φ implies the composition of convolutions in the model, f is the map from base space (x,y coordinate) to the input image and Ls implies the scale transformation. The following graph states the performance of different models on scale-MNIST classifcation task. 


|Model |Accuracy|Equivariance Error|
|---|---|---|
|Deformable Convolution|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/dcn_MNIST_acc.PNG" width="400">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/dcn_MNIST_equiv.PNG" width="400">|
|Deep Scale Space|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/DSS_MNIST_acc.PNG" width="400">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/DSS_MNIST_equiv.PNG" width="400">|
|SO2 Group equivariant CNN|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/so2_cnn_MNIST_acc.PNG" width="400">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/so2_cnn_MNIST_equiv.PNG" width="400">|

Remarks.
1. Deformable convolution is more effective than regular convolution in terms of both accuracy and equivariance.
2. The larger the scale transformation, the larger the equivariance error.
3. SO2 Group Equivariant CNN is the most effective in terms of equivariance but the computational cost of the model is too high to be practically applicable.

## 4. Result

1. Effect of hyperparameters on Deep Scale Space

The structure of the model is the same as ResNet50 where the convolutions are replaced with scale convolution modules.
Here base refers to the base of the gaussian kernel, interaction states the number of scale level considered during scale convolution and number of scale space refers to the number of levels in the scale dimension.  

The following table is to find out the effect of hyperparameters on scale-CIFAR10 classification task.

| Parameter | DSS v1 | Deform DSS | 
|---|---|---|
|Base|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v1_base.png" width="300">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v2_base.png" width="300">|
|Channel|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v1_channel.png" width="300">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v2_channel.png" width="300">|
|Init rate|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v1_init_rate.png" width="300">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v2_init_rate.png" width="300">|
|Interaction|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v1_interaction.png" width="300">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v2_interaction.png" width="300">|
|Number of scale space|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v1_n_scale.png" width="300">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/dss_param_mnist/dss_v2_n_scale.png" width="300">|

2. Result of new methods

<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/DSS2_MNIST_acc.PNG" width="350">   <img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/DSS2_MNIST_equiv.PNG" width="350">

In the graph, blue, green and orange each represent DSS v1, DSSv2 and deformable DSS.

As a remark on the results, we can see that deformable convolution works well when there is a small scale transformation but when there is large scale transformation, it seems to affect the performance. 

For the case of DSS v2, it generally underperforms than the original model. This could be due to the loss of information during the downscaling interpolation and the max pooling operation performed in order to match the dimensionality of the lifted feature maps.

I am still conducting hyperparameter tuning experiments and ensuring the validity of the code, so the results could be misleading. So the further results will be updated.

### Appendix

<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/1.PNG" width="800">
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/2.PNG" width="800" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/3.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/4.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/5.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/6.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/7.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/8.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/9.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/10.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/11.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/12.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/13.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/14.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/15.PNG" >
<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/img/16.PNG" >










