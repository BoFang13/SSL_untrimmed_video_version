import torch
import torch.nn as nn
from pick_models.pick_c3d import C3D
from models.net_part import *

class Pick(nn.Module):
    def __init__(self,base_network,num_options):

        super(Pick,self).__init__()

        self.base_network = base_network;

        self.fc = nn.Linear(512*3,512);

        self.relu = nn.ReLU(inplace=True);
        self.dropout = nn.Dropout(p=0.5)


        self.fc_classify = nn.Linear(512,num_options);




    def forward(self, x):
        nums = x.shape[1]
        f=[];
        for i in range(nums):
            clip = x[:,i,::];
            f.append(self.relu(self.base_network(clip)));



        f123 = torch.cat(((f[0],f[1],f[2])),dim=1);
        f12 = self.fc(f123)

        f12_dr = self.dropout(f12)
        out = self.fc_classify(f12_dr);



        return out;



if __name__ == '__main__':
    base=C3D(with_classifier=False);
    sscn=Pick(base,num_options=3);

    input_tensor = torch.autograd.Variable(torch.rand(2,3, 3, 16, 112, 112))
    #print(input_tensor)
    order = sscn(input_tensor)
    # m = nn.ConvTranspose3d(16,33,3,stride=2)
    # input_tensor = torch.autograd.Variable(torch.rand(20,16,10,50,100))
    # out = m(input_tensor)

    print(order.shape)

   # out = out[:,:,16,:,:]
