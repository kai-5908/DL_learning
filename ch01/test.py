from forward_net import TwoLayerNet
import numpy as np

x = np.random.randn(10,2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
