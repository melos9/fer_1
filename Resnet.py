import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


#Block内の最初のPlainアーキテクチャ。
class PlainA(chainer.Chain):
    def __init__(self, in_size, out_size, stride=2):
        super(PlainA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(in_size, out_size, 1, stride, 0, initialW=initialW)
            self.bn = L.BatchNormalization(out_size)
            self.conv2 = L.Convolution2D(out_size, out_size, 3, 1, 1, initialW=initialW)

    def __call__(self, x):
        h1 = F.relu(self.bn(self.conv1(x)))
        h1 = F.relu(self.bn(self.conv2(h1)))
        h1 = self.bn(self.conv2(h1))
        h2 = self.bn(self.conv1(x))

        return F.relu(h1 + h2)

#Block内の２番目からのPlainアーキテクチャ。
class PlainB(chainer.Chain):
    def __init__(self, ch):
        super(PlainB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():  
            self.conv = L.Convolution2D(ch, ch, 3, 1, 1, initialW=initialW)
            self.bn = L.BatchNormalization(ch)
            
    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        h = F.relu(self.bn(self.conv(h)))
        return F.relu(h + x)
        
class Block(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link('a', PlainA(in_size,out_size, stride))

        for i in range(1, layer):
            self.add_link('b{}'.format(i), PlainB(ch))
        
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)
        return h





class ResNet(chainer.Chain):
    #insize = 48
    def __init__(self,class_labels=9):
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res1 = Block(2,64,64,64,1)
            self.res2 = Block(2,64,128,128)
            self.res3 = Block(2,128,256,256)
            self.fc = L.Linear(256,class_labels)

    def __call__(self, x):

        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = F.average_pooling_2d(h, 3, stride=1)
        h = self.fc(h)

        return F.softmax(h)
