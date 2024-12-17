#
#学習していないもの
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


#モデル（インスタンス）を作る
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
image=image.unsqueeze(dim=2)

#モデルに入力画像を入れる
model.eval() #テストをしますの宣言（なくても可）
with torch.no_grad(): #勝手に学習しないように抑制
    logits=model(image)
    
print(logits)

#plt.bar(range(len(logits[0])),logits[0])
plt.show()

#ランダムな値が設定されている。学習していないので適当な値が帰ってくる

probs=logits.softmax(dim=1)
plt.bar(range(len(probs[0])),probs[0])
plt.ylim(0,1)
plt.show()

#もし正確な値を出すものだったら、一か所だけが１になるべき