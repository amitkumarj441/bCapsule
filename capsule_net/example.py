from collections import OrderedDict
from capsule_net import capsnet

p = CapsuleLayer(5, 3, 5, 3).named_parameters()
ps = OrderedDict((name, param) for (name, param) in p)
print(ps['weights'])
a = ps['weights'] * 8
print(a)

for i, n in enumerate(np.arange(5)):
    print(i, ' --- ', n)

for i in range(5):
    pass
print(i)
