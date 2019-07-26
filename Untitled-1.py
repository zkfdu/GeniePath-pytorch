import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
 
input_ = V(t.randn(2, 3))
print(input_)
model = nn.Linear(3, 4)
output1 = model(input_)
output2 = nn.functional.linear(input_, model.weight, model.bias)
print(output1 == output2)
 
b1 = nn.functional.relu(input_)
b2 = nn.ReLU()(input_)
print(b1 == b2)