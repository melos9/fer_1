import chainer
import Resnet,dataFromCsv
import numpy as np
import chainer.functions as F
from chainer import serializers,iterators,optimizer
from chainer.dataset import convert


#学習に関する基本情報の定義
EPOCH = 100 #学習回数
BATCH = 20 #バッチサイズ
NUM_SHAPE = 48 #画像一辺の長さ
LEARN_RATE = 0.001
WEIGHT_DECAY =1e-4
GPU = 0
SAVE_MODEL = "saved_model/myresnet.npz"

def learn(csvfile):

    train,publictest,_= dataFromCsv.dataFromCsv(csvfile)
  
    train_iter = iterators.SerialIterator(train,batch_size=BATCH,shuffle=True)
    publictest_iter = iterators.SerialIterator(publictest,batch_size=BATCH,repeat=False,shuffle=False)
    
    #学習したいモデルを決定、ただしモデルの出力はsoftmaxである必要がある
    model = Resnet.ResNet(class_labels=9)

    #GPU設定、GPUを使わない場合はコメントアウト可能
    chainer.cuda.get_device(GPU).use
    model.to_gpu()

    #最適化設定
    optimizer = chainer.optimizers.MomentumSGD(LEARN_RATE)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))

    #保存するモデル
    saved_model = model

    while train_iter.epoch < EPOCH:
        
        batch = train_iter.next()

        trainLossList = []
        
        x_array, y_array = convert.concat_examples(batch,GPU)

        x = chainer.Variable(x_array)
        y = chainer.Variable(y_array)
        m = model(x)

        loss_train = myCrossEntropyError(m,y)

        model.cleargrads()

        loss_train.backward()

        optimizer.update()

        trainLossList.append(chainer.cuda.to_cpu(loss_train.data))

        
        if  train_iter.is_new_epoch:
             
            testLossList = []
            
            #毎エポック後のmodelの精度を求め、publictest適用に置いて最良のmodelを保存
            for batch in publictest_iter:
                x_array, y_array = convert.concat_examples(batch, GPU)
                x = chainer.Variable(x_array)
                y = chainer.Variable(y_array)
                m = model(x)

                loss_test = myCrossEntropyError(m, y)
                testLossList.append(chainer.cuda.to_cpu(loss_test.data))

                
                if loss_test.data == np.min(testLossList):
                    saved_model = model
            
            publictest_iter.reset()
        
            print("epo:" + str(train_iter.epoch) + " train_loss:" + str(np.mean(trainLossList)) + " test_loss:" + str(np.mean(testLossList)))
            
    
 
    serializers.save_npz(SAVE_MODEL, saved_model)

    return


def myCrossEntropyError(m,y):
    DELTA = 1e-7 # マイナス無限大を発生させないように微小な値を追加する
    return -F.sum(y*F.log(m+DELTA)+(1-y)*F.log(1-m+DELTA))


if __name__ == "__main__":  
    
    learn("myferdata.csv")

