# Scale Equivariance
Analysis of equivariance under scale transformation in CNN.
For definitions and details relating to the models and ideas being incorporated, please refer to the Appendix presentation.

### Aim

To propose a convolution where scale equivariance is preserved to an extent while in class local transformation is captured by deformable kernel.
So the idea is to combine scale equivariant models' convolution with deformable convolution's offset prediction module for improved performanc.



### Equivariance in existing models

|Model |Accuracy|Equivariance Error|
|---|---|---|
|Deformable Convolution|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/dcn_MNIST_acc.PNG" width="400">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/dcn_MNIST_equiv.PNG" width="400">|
|Deep Scale Space|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/DSS_MNIST_acc.PNG" width="400">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/DSS_MNIST_equiv.PNG" width="400">|
|SO2 Group equivariant CNN|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/so2_cnn_MNIST_acc.PNG" width="400">|<img src="https://github.com/chanhopark00/scale_equivariance/blob/master/results/graph/so2_cnn_MNIST_equiv.PNG" width="400">|

### Result


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










