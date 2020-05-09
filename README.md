![https://fontmeme.com/permalink/200509/8f77639f1da69eb1a30d1edfceaf589c.png](https://fontmeme.com/permalink/200509/8f77639f1da69eb1a30d1edfceaf589c.png)

We all used the pretrained models by calling APIs which by default came with frameworks like Tensorflow, PyTorch, Keras etc. but what if you implement the whole architecture by your hand to gain the intution more clearly at every layer. It also helps one who wants to build custom/hybrid model architecture. Who knows that architecture may beat SOTA!

--------
So, My plan here is to implement the SOTA, old model Architectures from scratch. I am using  <b>Tensorflow</b> to code. If anyone wants to contribute in <b>PyTorch</b>, don't hesitate to create pull request I am waiting for that.
 
#### Current Implementation:
| Model | Link |
|-------|------|
| ResNet50 | [restnet50.py](https://github.com/niyazed/mfs/blob/master/models/resnet50/restnet50.py)|
| Unet | [unet.py](https://github.com/niyazed/mfs/blob/master/models/unet/unet.py)|
| Autoencoder Unet | [autoencoder_unet.py](https://github.com/niyazed/mfs/blob/master/models/hybrid/autoencoder_unet.py)|

