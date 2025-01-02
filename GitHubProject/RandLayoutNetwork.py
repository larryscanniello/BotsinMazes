import torch
from torch import nn

class RandLayoutNetwork(nn.Module):
  def __init__(self):
    super(RandLayoutNetwork, self).__init__()
    self.layoutconv = torch.nn.Conv3d( in_channels = 1, out_channels = 5, stride = (2,1,1) , bias = True, kernel_size=(2,3,3))
    self.botstateconv = torch.nn.Conv2d( in_channels = 1, out_channels = 5, stride = 1 , bias = True, kernel_size=(5,5))
    self.linear = torch.nn.Linear( in_features= 15300, out_features= 300, bias = True)
    self.linear_2 = torch.nn.Linear( in_features = 300, out_features = 5, bias = True )
  def forward(self, input_tensor):
    reshaped = torch.reshape( input_tensor, (-1,1,4,40,40) )
    layoutresult = self.layoutconv(reshaped[:,:,:2])
    highestprobindexresult = reshaped[:,:,2]
    botstateresult = self.botstateconv(reshaped[:,:,3])
    layoutresultflattened = torch.nn.Flatten()(layoutresult)
    destindexflattened = torch.nn.Flatten()(highestprobindexresult)
    botstateflattened = torch.nn.Flatten()(botstateresult)
    flattenedresult = torch.cat([layoutresultflattened,destindexflattened,botstateflattened],dim=1)
    flattenedresulttanh = torch.nn.Tanh()( flattenedresult )
    prelogits = self.linear( flattenedresulttanh )
    prelogits = torch.nn.ELU()(prelogits)   
    logits = self.linear_2( prelogits )
    probabilities = torch.nn.Softmax( dim = 1 )( logits )
    return logits, probabilities