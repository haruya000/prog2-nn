import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models
#データセットの前処理を定義
#（Transformsでダウンロードするときに変換している 何回か使うので変数にいれる）
ds_transform=transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32,scale=True)])

#データセットの読み込み
ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

ds_test=datasets.FashionMNIST(
    root='data',
    train='False',
    download=True, #テスト用データセット
    transform=ds_transform
)

# それぞれにデータローダーを作る
# バッチサイズ（分けるサイズ）をきめる


#学習用
batchnosize=64
dataloader_train=torch.utils.data.DataLoader(
    ds_train,
    batch_size=batchnosize,
    shuffle=True,
    )
#テスト用。シャッフルしない。（shuffleを書かなくてもFalseになる)
dataloader_test=torch.utils.data.DataLoader(
    ds_test,
    batch_size=batchnosize,
    shuffle=False,
    )

for image_batch,label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break
#クラス確立で１にするのではなくロジック確立で求めることになる

model=models.model()
#精度（正解率を計算）を計算
acc_train=models.test_accuracy(model,dataloader_train)
print(f'train accuracy:{acc_train*100:.3f}%')
acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accuracy:{acc_test*100:.4f}%')

#ロス関数（いろいろあるがどの関数も０に一番近くなる（小さくなる）形がベスト）
loss_fn=torch.nn.CrossEntropyLoss()
#最適化手法の選択
learning_rate=0.003
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
#criterion(規準)とも呼ぶ

#精度計算
acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accuracy:{acc_test*100:.3f}%')
#学習
models.train(model,dataloader_test,loss_fn,optimizer)
#もう一度制度計算
acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accuracy:{acc_test*100:.3f}%')