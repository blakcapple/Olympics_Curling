import torch.nn as nn 
import torch
import torch.nn.functional as F

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):

    out_channel = 32
    hidden_size = 256
    kernel_size = 3
    stride = 1
    use_Relu = True
    use_orthogonal = True
    
    def __init__(self, state_shape):
        
        super().__init__()
        active_func = [nn.Tanh(), nn.ReLU()][self.use_Relu]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_Relu])
        input_channel = state_shape[0]
        input_width = state_shape[1]
        input_height = state_shape[2]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.block = Residual_Block(input_channel, self.out_channel)
        self.flatten = Flatten()
        with torch.no_grad():
            dummy_ob = torch.ones(1, input_channel, input_width, input_height).float()
            cnn_out = self.block(dummy_ob)
            n_flatten = self.flatten(cnn_out).shape[1]
        self.linear = nn.Sequential(init_(nn.Linear(n_flatten, self.hidden_size)), nn.ReLU())
    def forward(self, input):
        
        cnn_output = self.block(input)
        cnn_flatten = self.flatten(cnn_output)
        output = self.linear(cnn_flatten)
        return output

class Residual_Block(nn.Module):

    def __init__(self, inplanes, planes, stride=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                        padding=1, bias = False) # conv2d 和conv3d 的区别
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                        padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual 
        out = F.relu(out)

        return out 