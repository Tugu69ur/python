import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nChannels, nClasses):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nChannels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        self.rnn = nn.LSTM(128*(imgH//4), 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, nClasses)

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W]
        b,c,h,w = x.size()
        x = x.permute(0,3,1,2).contiguous().view(b,w,c*h)  # [B, W, C*H]
        x,_ = self.rnn(x)
        x = self.fc(x)
        x = x.log_softmax(2)
        return x
