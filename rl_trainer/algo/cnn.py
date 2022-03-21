import torch.nn as nn 
import torch

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
        # cnn1_out_shape = [input_width - self.kernel_size + self.stride, input_height - self.kernel_size + self.stride]
        # pool_out_shape = [int((cnn1_out_shape[0] - 2)/2) + 1, int((cnn1_out_shape[0] - 2)/2) + 1 ]
        # cnn2_out_shape = [pool_out_shape[0] - self.kernel_size + self.stride, pool_out_shape[1] - self.kernel_size + self.stride]
        # # cnn_out_size = cnn2_out_shape[0] * cnn2_out_shape[1] * self.out_channel
        pool = nn.AvgPool2d(kernel_size=2)
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=self.stride)
                  ),
            pool,
            init_(nn.Conv2d(in_channels=self.out_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=self.stride)),
            active_func,
            Flatten(),
                            )
        with torch.no_grad():
            dummy_ob = torch.ones(1, input_channel, input_width, input_height).float()
            n_flatten = self.cnn(dummy_ob).shape[1] # 6400
        self.linear = nn.Sequential(init_(nn.Linear(n_flatten, self.hidden_size)), nn.ReLU())
    def forward(self, input):
        cnn_output = self.cnn(input)
        output = self.linear(cnn_output)
        return output