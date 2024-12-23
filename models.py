import torch
from torch import nn

class model(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)

        )
    def forward(self,x): #計算を行っている
        x=self.flatten(x)
        logits=self.network(x)
        return logits
    
#predict_batchを見れば１行の値がわかる
#log

# array=tensor
# .sum=Trueの数
#.item=tensorでなく値になる
def test_accuracy(model,dataloader,device='cpu'):
    n_corrects=0 #正解の個数

    model.to(device)

    model.eval()
    for image_batch,label_batch in dataloader:
        image_batch.to(device)
        label_batch.to(device)
        with torch.no_grad():
            #logitsからどれだけ目的の数からうまくいかなかった度合い（誤差）も調べる
            #もしくはこうなるべきというのがあるのなら目的関数を作る
            logits_batch=model(image_batch)
            
        predict_batch=logits_batch.argmax(dim=1)
        n_corrects+=(label_batch==predict_batch).sum().item()
#精度（正解率を計算する)（全体からどれだけ正解しているのか）
    accuracy=n_corrects/len(dataloader.dataset)

    return accuracy

def train(model,dataloader,loss_fn,optimizer,device='cpu'):
    #1epochの学習を行う
    model.to(device)
    model.train() #最適化していくのでnogradを消す
    for image_batch,label_batch in dataloader:
        image_batch=image_batch.to(device)
        label_batch=label_batch.to(device)
        #モデルにバッチを入れて計算
        logits_batch=model(image_batch)
        
        #損失（誤差）を計算する
        loss=loss_fn(logits_batch,label_batch)

        #最適化
        optimizer.zero_grad()
        loss.backward() #誤差逆伝播法(back propagation)
        optimizer.step() #一個上の結果を使ってパラメーターをちょっと動かす
    #最後のパッチのロス
    return loss.item()

def test(model,dataloader,loss_fn,device='cpu'):
    loss_total=0.0
    model=model.to(device)
    model.eval()

    for image_batch,label_batch in dataloader:
        image_batch.to(device)
        label_batch.to(device)
        with torch.no_grad():
            logits_batch=model(image_batch)
        loss=loss_fn(logits_batch,label_batch)
        loss_total+=loss.item()

    return loss_total/len(dataloader)

