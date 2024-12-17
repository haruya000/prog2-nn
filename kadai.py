import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

model=models.model()
print(model)

ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    #Transformsでダウンロードするときに変換している
    transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32,scale=True)
    ])
)
#imageは0-1に正規化されたTensor(1,28,28)
image,target=ds_train[0]
#(1,28,28)=>(1,1,28,28)
image=image.unsqueeze(dim=0)

#モデルに入力画像を入れる
model.eval()
with torch.no_grad():
    logits=model(image)
    
print(logits)

plt.show()


probs=logits.softmax(dim=1)
#plt.bar(range(len(probs[0])),probs[0])
#plt.ylim(0,1)
#plt.show()

plt.subplot(1,2,1)
plt.imshow(image[0,0],cmap='gray_r')
plt.title(f'class:{target}({datasets.FashionMNIST.classes[target]})')

plt.subplot(1,2,2)
plt.bar(range(len(probs[0])),probs[0])
plt.ylim(0,1)
plt.title(f'predicted class :{probs[0].argmax()}')
plt.show()