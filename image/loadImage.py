import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from PIL import Image  
Image.MAX_IMAGE_PIXELS = None

minDist=int(13/0.3)

def loadandsplit(samplename,imagedir,diamThresh,val,test,ifFlip=True,minCutoff=6,seed=3):
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images','pi_sum.tif')
    image=mpimg.imread(imagepath).copy()
    
    if samplename=='AD_mouse9494':
        image=np.flipud(image)
    elif samplename=='AD_mouse9498':
        image=np.fliplr(image)
    elif samplename=='AD_mouse9735':
        image=np.fliplr(image)
        image=np.flipud(image)
    
    rowSplits=int(np.floor(image.shape[0]/diamThresh)+((image.shape[0]%diamThresh)>(minDist*minCutoff)))
    colSplits=int(np.floor(image.shape[1]/diamThresh)+((image.shape[1]%diamThresh)>(minDist*minCutoff)))
    res=np.zeros((rowSplits*colSplits,1,diamThresh,diamThresh))
    
    for r in range(rowSplits):
        for c in range(colSplits):
            imagerc=image[r*diamThresh:min((r+1)*diamThresh,image.shape[0]),c*diamThresh:min((c+1)*diamThresh,image.shape[1])]
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
            res[r*colSplits+c,0,:imagerc.shape[0],:imagerc.shape[1]]=imagerc
    imTrain,imValTest=train_test_split(res,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest=train_test_split(imValTest,test_size=test/(val+test), random_state=seed, shuffle=True)
    return imTrain,imVal,imTest