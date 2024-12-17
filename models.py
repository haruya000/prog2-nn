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