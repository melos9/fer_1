import chainer
import Resnet,dataFromCsv
import numpy as np
from chainer.dataset import convert 
from learn import myCrossEntropyError

#学習に関する基本情報の定義
CLASS_LIST = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","unknown"]
GPU = 0

def test(csvfile,m):

    _,_,privatetest = dataFromCsv.dataFromCsv(csvfile)
    model = Resnet.ResNet()

    chainer.serializers.load_npz("saved_model/myresnet.npz",model)
    chainer.cuda.get_device(0).use() 
    model.to_gpu()

    x_array, y_array = convert.concat_examples(privatetest,GPU)
 
    #memoryに乗り切らないので最初の100の画像についてtest
    x = chainer.Variable(x_array[0:100,:,:])
    y = chainer.Variable(y_array[0:100,:])

    m = model(x)

    for (mm,yy) in zip(m,y):
        for i in range(len(CLASS_LIST)):
            print("{0:<10} ... pred:{1:<15} ans:{2:<5}".format(CLASS_LIST[i] ,str(mm[i].data),str(yy[i].data)))

        print("------------------------------")
if __name__ == "__main__":  
    
    test("myferdata.csv","saved_model/myresnet.npz")