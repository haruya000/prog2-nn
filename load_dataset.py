import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import torch

ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    )

print(f'num of datasets:{len(ds_train)}')

image,target=ds_train[988]
print(type(image),target)
plt.imshow(image,cmap='gray_r')
plt.title(target)
plt.show()

image_tensor=transforms.functional.to_image(image)
image_tensor=transforms.functional.to_dtype(image_tensor,dtype=torch.float32,scale=True)
#将来廃止される方法
# image=transforms.functional.to_tensor(image)
print(image_tensor.shape,image_tensor.dtype)
print(image_tensor.min(),image_tensor.max())
#for i in range(5):
    #for j in range(5):
        #k=i*5+j
        #image,target=ds_train[k]
        #plt.subplot(5,5,k+1)
        #plt.imshow(image,cmap='gray_r')
#plt.show()