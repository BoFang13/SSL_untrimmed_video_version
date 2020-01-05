import torch
import sys
sys.path.append('..')
from model.c3d import C3D
from model.net_part import *


class SSLNET(nn.Module):
    def __init__(self,
                 base_network,
                 with_classifier=False,
                 num_classes=8):
        super(SSLNET, self).__init__()

        self.base_network = base_network
        self.with_classifier = with_classifier
        self.num_classes = num_classes

        self.pool = nn.AdaptiveAvgPool3d(1)
        if with_classifier:
            self.fc6 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.base_network(x)
        x = self.pool(x)
        x = x.view(-1, 512)

        if self.with_classifier:
            x = self.fc6(x)

        return x


if __name__ == '__main__':
    base = C3D(with_classifier=False)
    ssl_net = SSLNET(base, with_classifier=True, num_classes=8)

    input_tensor = torch.autograd.Variable(torch.rand(8, 3, 16, 112, 112))

    output_tensor = ssl_net(input_tensor)
    print(output_tensor.shape)