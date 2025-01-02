import torch
from torch import nn

class StochMouseNetwork(nn.Module):
  def __init__(self):
    super(StochMouseNetwork, self).__init__()
    self.conv_1 = torch.nn.Conv3d( in_channels = 1, out_channels = 40, stride = 1 , bias = True, kernel_size=(1,40,40))
    self.linear = torch.nn.Linear( in_features= 40*4, out_features= 300, bias = True)
    self.linear_2 = torch.nn.Linear( in_features = 300, out_features = 5, bias = True )
  def forward(self, input_tensor):
    reshaped = torch.reshape( input_tensor, (-1,1,4,40,40) )
    conv_results = self.conv_1( reshaped )
    conv_results = torch.nn.Tanh()( conv_results )
    flattened = torch.nn.Flatten()( conv_results )
    prelogits = self.linear( flattened )
    prelogits = torch.nn.ELU()(prelogits)
    logits = self.linear_2( prelogits )
    probabilities = torch.nn.Softmax( dim = 1 )( logits )
    return logits, probabilities