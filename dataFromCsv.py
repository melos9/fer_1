import numpy as np
import pandas as pd
import cv2
from chainer.datasets import tuple_dataset
from random import getrandbits 

#学習に関する基本情報の定義
NUM_SHAPE = 48 #画像一辺の長さ
TRAIN_DATA_SIZE_MAG = 2 #水増しで元のデータサイズの何倍の量まで増やすか


#Csvファイルから画像とラベルを読み込む
def dataFromCsv(csvfile):

    data = pd.read_csv(csvfile,delimiter=',')

    train_data = data[data['Usage']=='Training']
    publictest_data = data[data['Usage']=='PublicTest']
    privatetest_data = data[data['Usage']=='PrivateTest']

    #1行のデータを画像のカタチにする(画像枚数、1、縦、横)
    train_x = pixelsToArray_x(train_data) 
    publictest_x = pixelsToArray_x(publictest_data)
    privatetest_x = pixelsToArray_x(privatetest_data)

    #ラベルは["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","unknown,NA"]
    #NA以外をyに入れる
    #各画像へのラベルは合計10になるので、10で割って0-1にする
    train_y = np.array(train_data.iloc[:,2:11],dtype=np.float32)/10
    publictest_y = np.array(publictest_data.iloc[:,2:11],dtype=np.float32)/10
    privatetest_y = np.array(privatetest_data.iloc[:,2:11],dtype=np.float32)/10

    #水増し
    train_x,train_y = augmentation(train_x,train_y)

    #tuple化
    train = tuple_dataset.TupleDataset(train_x,train_y)
    publictest = tuple_dataset.TupleDataset(publictest_x,publictest_y)
    privatetest = tuple_dataset.TupleDataset(privatetest_x,privatetest_y)
    
    return train,publictest,privatetest


#水増し(holizontal Flip,Scale augmentation)
def augmentation(x_array,y_array,train_data_size_mag = TRAIN_DATA_SIZE_MAG): 

    #データ変換の処理4つ
    #関数が適用されるかはランダム
    def normalization(img):
        return (img - np.mean(img))/np.std(img)
    
    def gausianNoise(img):
        MEAN = 0
        SIGMA = 15

        gaussfilter = np.random.normal(MEAN,SIGMA,(img.shape))
        return img + gaussfilter
          
    def holizontalFlip(img):
        return img[:,::-1]
        
    def scaleAugmentation(img):
        SCALE_MIN = 50
        SCALE_MAX = 80

        #拡大処理、入力された画像サイズ48*48に対して、50*50~80*80まで拡大
        SCALE_SIZE = np.random.randint(SCALE_MIN,SCALE_MAX)

        #リサイズ
        scale_img = cv2.resize(img,(SCALE_SIZE,SCALE_SIZE))
      
        top = np.random.randint(0,SCALE_SIZE-NUM_SHAPE)
        left = np.random.randint(0,SCALE_SIZE-NUM_SHAPE)
        bottom = top + NUM_SHAPE
        right = left + NUM_SHAPE

        return scale_img[top:bottom,left:right]
    

    def activateAugmentFforArray(f,x_array,activateP):
        
       #変換用関数fを画像に適用させるかどうかをランダムに決める
        def randActivateF(f,img):
            if np.random.rand()>activateP:
                return img 
            return f(img) 

        imglist = []
        #x_arrayは[データ数,色数,縦,横]なので2回ループして画像毎の関数を(ランダムに)適用
        for imgC in x_array:
            imglist.append([randActivateF(f,img) for img in imgC])
                
        return np.array(imglist)

    #変換処理対象データをtrain_data_size_mag-1用意(1セットは元の画像にするため-1)
    changed_x_array = np.concatenate([x_array]*(train_data_size_mag-1),axis=0)

    #変換の種類ごとにactivateAugmentFforArrayを適用して、画像の変換（もしくは無変換）を行う
    changed_x_array = activateAugmentFforArray(normalization,changed_x_array,0.2)
    changed_x_array = activateAugmentFforArray(gausianNoise,changed_x_array,0.2)
    changed_x_array = activateAugmentFforArray(holizontalFlip,changed_x_array,1)
    changed_x_array = activateAugmentFforArray(scaleAugmentation,changed_x_array,0.2)

    return np.concatenate([x_array,changed_x_array],axis=0).astype(np.float32),np.concatenate([y_array]*train_data_size_mag,axis=0)

#1行のデータを画像の形にする
def pixelsToArray_x(data):
    np_x = np.array([np.fromstring(image,np.float32,sep=' ')/255 for image in np.array(data['pixels'])])
    np_x.shape =(np_x.shape[0],1,NUM_SHAPE,NUM_SHAPE)
    return np_x
