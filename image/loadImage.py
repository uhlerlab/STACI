import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image  
Image.MAX_IMAGE_PIXELS = None
from scipy import stats
from skimage import io

minDist=int(13/0.3)

def loadandsplit(samplename,imagedir,diamThresh,overlap,val,test,ifFlip=True,minCutoff=6,seed=3,split=True,imagename='pi_sum.tif',minmaxscale=True,nchannels=1,clf=False,returnPos=False):
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images',imagename)
    image=mpimg.imread(imagepath,'tif').copy()
    print(image.shape)
    if len(image.shape)==3:
        image=image[:,:,0]
    
    if ifFlip:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    stride=diamThresh-overlap
    rowSplits=int(np.floor(image.shape[0]/stride)+((image.shape[0]%stride-overlap)>(minDist*minCutoff)))
    colSplits=int(np.floor(image.shape[1]/stride)+((image.shape[1]%stride-overlap)>(minDist*minCutoff)))
    res=np.zeros((rowSplits*colSplits,nchannels,diamThresh,diamThresh))
    coordNeg=np.zeros((rowSplits*colSplits,2))
    
    for r in range(rowSplits):
        for c in range(colSplits):
            imagerc=image[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
            res[r*colSplits+c,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[r*colSplits+c]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])

#     image=None
    if not split:
        if not returnPos:
            return res,rowSplits,colSplits
        else:
            return res,coordNeg,rowSplits,colSplits
    imTrain,imValTest,coordTrain,coordNegValTest=train_test_split(res,coordNeg,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest,coordVal,coordTest=train_test_split(imValTest,coordNegValTest,test_size=test/(val+test), random_state=seed, shuffle=True)
    if clf:
        if not returnPos:
            return imTrain,imVal,imTest,np.zeros(imTrain.shape[0]),np.zeros(imVal.shape[0]),np.zeros(imTest.shape[0])
        else:
            return imTrain,imVal,imTest,np.zeros(imTrain.shape[0]),np.zeros(imVal.shape[0]),np.zeros(imTest.shape[0]),coordTrain,coordVal,coordTest
    else:
        return imTrain,imVal,imTest
    
def load_cellCentroid_plaque(plaqueImageName,coord,samplename,imagedir,diamThresh,ifFlip=False,seed=3,imagename='pi_sum.tif',minmaxscale=True,nchannels=1):
    np.random.seed(seed)
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images',imagename)
    image=mpimg.imread(imagepath,'tif')
    
    if plaqueImageName:
        plaqueImagepath=os.path.join(imagedir,samplename,'trimmed_images',plaqueImageName)
        plaqueImage=mpimg.imread(plaqueImagepath,'tif')[:,:,0]
    
    print(image.shape)
    if len(image.shape)==3:
        image=image[:,:,0]
            
    if ifFlip==True:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    res=np.zeros((coord.shape[0],nchannels,diamThresh,diamThresh))
    labelsRes=np.zeros(coord.shape[0])
    radius=int(diamThresh/2)
    for c in range(coord.shape[0]):
        centroid=coord[c]
        if centroid[0]>=image.shape[0]:
            print('row '+str(centroid[0]))
        if centroid[1]>=image.shape[1]:
            print('col '+str(centroid[1]))
        rowstart=max(0,centroid[0]-radius)
        rowEnd=min(image.shape[0],centroid[0]+radius)
        colstart=max(0,centroid[1]-radius)
        colEnd=min(image.shape[1],centroid[1]+radius)
        imagerc=image[rowstart:rowEnd,colstart:colEnd]
        if minmaxscale:
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            if imagercmin==imagercmax:
                print('no cells')
                imagerc=np.zeros_like(imagerc)
            else:
                imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
        res[c,:,:(rowEnd-rowstart),:(colEnd-colstart)]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
        if plaqueImageName:
            labelsRes[c]=np.sum(plaqueImage[rowstart:rowEnd,colstart:colEnd]>0)
#     image=None
    if ifFlip=='randomize':
        groupsize=int(res.shape[0]/8)
        rdmIdx=np.arange(res.shape[0])
        np.random.shuffle(rdmIdx)
        res[:groupsize]=np.flip(res[:groupsize],axis=3)
        res[rdmIdx[groupsize:groupsize*2]]=np.flip(res[rdmIdx[groupsize:groupsize*2]],axis=2)
        res[rdmIdx[groupsize*2:groupsize*3]]=np.flip(res[rdmIdx[groupsize*2:groupsize*3]],axis=(2,3))
        res[rdmIdx[groupsize*3:groupsize*4]]=np.rot90(res[rdmIdx[groupsize*3:groupsize*4]],axes=(2,3))
        res[rdmIdx[groupsize*4:groupsize*5]]=np.flip(np.rot90(res[rdmIdx[groupsize*4:groupsize*5]],axes=(2,3)),axis=2)
        res[rdmIdx[groupsize*5:groupsize*6]]=np.flip(np.rot90(res[rdmIdx[groupsize*5:groupsize*6]],axes=(2,3)),axis=3)
        res[rdmIdx[groupsize*6:groupsize*7]]=np.flip(np.rot90(res[rdmIdx[groupsize*6:groupsize*7]],axes=(2,3)),axis=(2,3))
    return res,labelsRes

def load_cellCentroid(coord,samplename,imagedir,diamThresh,ifFlip=False,seed=3,imagename='pi_sum.tif',minmaxscale=True,nchannels=1,addDir='/trimmed_images/'):
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename+addDir+imagename)
#     image=mpimg.imread(imagepath,'tif').copy()
    image=io.imread(imagepath)
    print(image.shape)
    if len(image.shape)==3:
        image=image[:,:,0]
    
    if ifFlip:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    res=np.zeros((coord.shape[0],nchannels,diamThresh,diamThresh))
    radius=int(diamThresh/2)
    for c in range(coord.shape[0]):
        centroid=coord[c]
        if centroid[0]>=image.shape[0]:
            print('row '+str(centroid[0]))
        if centroid[1]>=image.shape[1]:
            print('col '+str(centroid[1]))
        rowstart=max(0,centroid[0]-radius)
        rowEnd=min(image.shape[0],centroid[0]+radius)
        colstart=max(0,centroid[1]-radius)
        colEnd=min(image.shape[1],centroid[1]+radius)
        imagerc=image[rowstart:rowEnd,colstart:colEnd]
        if minmaxscale:
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            if imagercmin==imagercmax:
                print('no cells')
                imagerc=np.zeros_like(imagerc)
            else:
                imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
        res[c,:,:(rowEnd-rowstart),:(colEnd-colstart)]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
#     image=None
    return res
    
def loadandsplitPlaque(coord,cutoffradius,samplename,imagedir,diamThresh,overlap,val,test,ifFlip=False,minCutoff=6,seed=3,split=True,imagename='pi_sum.tif',minmaxscale=True,nchannels=1,returnPos=False):
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images',imagename)
    image=mpimg.imread(imagepath,'tif').copy()
    print(image.shape)
    
    if ifFlip:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    #plaque images
    plaqueRes=np.zeros((coord.shape[0],nchannels,diamThresh,diamThresh))
    plaqueBinary=np.zeros_like(image)
    radius=int(diamThresh/2)
    for c in range(coord.shape[0]):
        centroid=coord[c]
        if centroid[0]>=image.shape[0]:
            print('row '+str(centroid[0]))
        if centroid[1]>=image.shape[1]:
            print('col '+str(centroid[1]))
        rowstart=max(0,centroid[0]-radius)
        rowEnd=min(image.shape[0],centroid[0]+radius)
        colstart=max(0,centroid[1]-radius)
        colEnd=min(image.shape[1],centroid[1]+radius)
        imagerc=image[rowstart:rowEnd,colstart:colEnd]
        if minmaxscale:
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            if imagercmin==imagercmax:
                print('no cells')
                imagerc=np.zeros_like(imagerc)
            else:
                imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
        plaqueRes[c,:,:(rowEnd-rowstart),:(colEnd-colstart)]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
        plaqueBinary[max(0,centroid[0]-cutoffradius):min(image.shape[0],centroid[0]+cutoffradius),max(0,centroid[1]-cutoffradius):min(image.shape[1],centroid[1]+cutoffradius)]=1
    
    stride=diamThresh-overlap
    rowSplits=int(np.floor(image.shape[0]/stride)+((image.shape[0]%stride-overlap)>(minDist*minCutoff)))
    colSplits=int(np.floor(image.shape[1]/stride)+((image.shape[1]%stride-overlap)>(minDist*minCutoff)))
    res=np.zeros((rowSplits*colSplits,nchannels,diamThresh,diamThresh))
    coordNeg=np.zeros((rowSplits*colSplits,2))
    
    for r in range(rowSplits):
        for c in range(colSplits):
            if np.sum(plaqueBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])])>0:
                continue
            imagerc=image[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]
            plaqueBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]=1
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
            res[r*colSplits+c,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[r*colSplits+c]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
    residx=(np.sum(res,axis=(1,2,3))>0)
    print('no plaque '+str(np.sum(residx)))
    res=res[residx]
    coordNeg=coordNeg[residx]
    resAll=np.concatenate((res,plaqueRes),axis=0)
    coordAll=np.concatenate((coordNeg,coord),axis=0)
    labels=np.concatenate((np.repeat(0,res.shape[0]),np.repeat(1,plaqueRes.shape[0])))
            
    if not split:
        return resAll,labels,rowSplits,colSplits
    imTrain,imValTest,labelsTrain,labelsValTest,coordTrain,coordValTest=train_test_split(resAll,labels,coordAll,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest,labelsVal,labelsTest,coordVal,coordTest=train_test_split(imValTest,labelsValTest,coordValTest,test_size=test/(val+test), random_state=seed, shuffle=True)
    if not returnPos:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest
    else:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest,coordTrain,coordVal,coordTest
    
def loadandsplitPlaque_overlap(areaThresh,coord,cutoffradius,samplename,imagedir,diamThresh,overlap,val,test,ifFlip=False,minCutoff=6,seed=3,split=True,imagename='pi_sum.tif',minmaxscale=True,nchannels=1,returnPos=False):
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images',imagename)
    image=mpimg.imread(imagepath,'tif').copy()
    print(image.shape)
    
    if ifFlip:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    #plaque images
    plaqueRes=np.zeros((coord.shape[0],nchannels,diamThresh,diamThresh))
    plaqueBinary=np.zeros_like(image)
    radius=int(diamThresh/2)
    for c in range(coord.shape[0]):
        centroid=coord[c]
        if centroid[0]>=image.shape[0]:
            print('row '+str(centroid[0]))
        if centroid[1]>=image.shape[1]:
            print('col '+str(centroid[1]))
        rowstart=max(0,centroid[0]-radius)
        rowEnd=min(image.shape[0],centroid[0]+radius)
        colstart=max(0,centroid[1]-radius)
        colEnd=min(image.shape[1],centroid[1]+radius)
        imagerc=image[rowstart:rowEnd,colstart:colEnd]
        if minmaxscale:
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            if imagercmin==imagercmax:
                print('no cells')
                imagerc=np.zeros_like(imagerc)
            else:
                imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
        plaqueRes[c,:,:(rowEnd-rowstart),:(colEnd-colstart)]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
        plaqueBinary[max(0,centroid[0]-cutoffradius):min(image.shape[0],centroid[0]+cutoffradius),max(0,centroid[1]-cutoffradius):min(image.shape[1],centroid[1]+cutoffradius)]=1
    plaqueCentroidBinary=np.copy(plaqueBinary)
    
    stride=diamThresh-overlap
    rowSplits=int(np.floor(image.shape[0]/stride)+((image.shape[0]%stride-overlap)>(minDist*minCutoff)))
    colSplits=int(np.floor(image.shape[1]/stride)+((image.shape[1]%stride-overlap)>(minDist*minCutoff)))
    res=np.zeros((rowSplits*colSplits,nchannels,diamThresh,diamThresh))
    coordNeg=np.zeros((rowSplits*colSplits,2))
    
    idx=0
    for r in range(rowSplits):
        for c in range(colSplits):
            if np.sum(plaqueCentroidBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]), c*stride:min((c+1)*stride+overlap,image.shape[1])])<areaThresh:
                continue
            imagerc=image[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]
            plaqueBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]=1
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
#             imagerc=np.pad(imagerc,((0,diamThresh-imagerc.shape[0]),(0,diamThresh-imagerc.shape[1])))
            res[idx,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[idx]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
            idx+=1
#             plaqueRes=np.concatenate((plaqueRes,imagerc.reshape((1,nchannels,imagerc.shape[0],imagerc.shape[1]))),axis=0)
#             coord=np.concatenate((coord,np.array([r*stride+diamThresh/2,c*stride+diamThresh/2]).reshape((1,2))),axis=0)
#     plaqueRes=np.concatenate((res[:idx],plaqueRes),axis=0)
#     coord=np.concatenate((coordNeg[:idx],coord),axis=0)
    labels=np.repeat(1,coord.shape[0]+idx)
    naddedplaque=idx
    print('plaque'+str(labels.shape[0]))
    
    for r in range(rowSplits):
        for c in range(colSplits):
            if np.sum(plaqueBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]), c*stride:min((c+1)*stride+overlap,image.shape[1])])>0:
                continue
            imagerc=image[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]
#             plaqueBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]=1
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
#             imagerc=np.pad(imagerc,((0,diamThresh-imagerc.shape[0]),(0,diamThresh-imagerc.shape[1])))
            res[idx,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[idx]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
            idx+=1
#             plaqueRes=np.concatenate((plaqueRes,imagerc.reshape((1,nchannels,imagerc.shape[0],imagerc.shape[1]))),axis=0)
#             coord=np.concatenate((coord,np.array([r*stride+diamThresh/2,c*stride+diamThresh/2]).reshape((1,2))),axis=0)
    plaqueRes=np.concatenate((plaqueRes,res[:idx]),axis=0)
    coord=np.concatenate((coord,coordNeg[:idx]),axis=0)
    labels=np.concatenate((labels,np.repeat(0,idx-naddedplaque)))
    print('no plaque'+str(idx-naddedplaque))
    
    res=None
    coorNeg=None
    plaqueCentroidBinary=None
    plaqueBinary=None
    resAll=plaqueRes
    coordAll=coord
            
    if not split:
        return resAll,labels,rowSplits,colSplits
    imTrain,imValTest,labelsTrain,labelsValTest,coordTrain,coordValTest=train_test_split(resAll,labels,coordAll,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest,labelsVal,labelsTest,coordVal,coordTest=train_test_split(imValTest,labelsValTest,coordValTest,test_size=test/(val+test), random_state=seed, shuffle=True)
    if not returnPos:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest
    else:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest,coordTrain,coordVal,coordTest
    

def loadandsplitPlaque_overlap_regrs(plaqueImageName,plaqueSizeFactor,areaThresh,coord,cutoffradius,samplename,imagedir,diamThresh,overlap,val,test,ifFlip=False,minCutoff=6,seed=3,split=True,imagename='pi_sum.tif',minmaxscale=True,nchannels=1,returnPos=False):
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images',imagename)
    image=mpimg.imread(imagepath,'tif').copy()
    
    plaqueImagepath=os.path.join(imagedir,samplename,'trimmed_images',plaqueImageName)
    plaqueImage=mpimg.imread(plaqueImagepath,'tif')[:,:,0]
#     print(np.sum(plaqueImage[:,:,0]>0))
    
    if ifFlip:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    #plaque images
    labelsRes=np.zeros(coord.shape[0])
    plaqueRes=np.zeros((coord.shape[0],nchannels,diamThresh,diamThresh))
    plaqueBinary=np.zeros_like(image)
    radius=int(diamThresh/2)
    for c in range(coord.shape[0]):
        centroid=coord[c]
        if centroid[0]>=image.shape[0]:
            print('row '+str(centroid[0]))
        if centroid[1]>=image.shape[1]:
            print('col '+str(centroid[1]))
        rowstart=max(0,centroid[0]-radius)
        rowend=min(image.shape[0],centroid[0]+radius)
        colstart=max(0,centroid[1]-radius)
        colend=min(image.shape[1],centroid[1]+radius)
        imagerc=image[rowstart:rowend,colstart:colend]
#         print(imagerc.shape)
        if minmaxscale:
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            if imagercmin==imagercmax:
                print('no cells')
                imagerc=np.zeros_like(imagerc)
            else:
                imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
        plaqueRes[c,:,:(rowend-rowstart),:(colend-colstart)]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
        plaqueBinary[max(0,centroid[0]-cutoffradius):min(image.shape[0],centroid[0]+cutoffradius),max(0,centroid[1]-cutoffradius):min(image.shape[1],centroid[1]+cutoffradius)]=1
        labelsRes[c]=np.sum(plaqueImage[rowstart:rowend,colstart:colend]>0)/plaqueSizeFactor
#         print((plaqueImage[rowstart:rowend,colstart:colend]>0).shape)
#         print((rowend-rowstart)*(colend-colstart))
#         labelsRes[c]=labelsRes[c]/(rowend-rowstart)/(colend-colstart)
    plaqueCentroidBinary=np.copy(plaqueBinary)
    
    stride=diamThresh-overlap
    rowSplits=int(np.floor(image.shape[0]/stride)+((image.shape[0]%stride-overlap)>(minDist*minCutoff)))
    colSplits=int(np.floor(image.shape[1]/stride)+((image.shape[1]%stride-overlap)>(minDist*minCutoff)))
    res=np.zeros((rowSplits*colSplits,nchannels,diamThresh,diamThresh))
    coordNeg=np.zeros((rowSplits*colSplits,2))
    labels=np.zeros(rowSplits*colSplits)
    #print(rowSplits*colSplits)
    
    idx=0
    for r in range(rowSplits):
        for c in range(colSplits):
            rowstart=r*stride
            rowend=min((r+1)*stride+overlap,image.shape[0])
            colstart=c*stride
            colend=min((c+1)*stride+overlap,image.shape[1])
            if np.sum(plaqueCentroidBinary[rowstart:rowend, colstart:colend])<areaThresh:
                continue
            imagerc=image[rowstart:rowend,colstart:colend]
#             print(imagerc.shape)
            plaqueBinary[rowstart:rowend,colstart:colend]=1
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
#             imagerc=np.pad(imagerc,((0,diamThresh-imagerc.shape[0]),(0,diamThresh-imagerc.shape[1])))
            res[idx,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[idx]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
            labels[idx] = np.sum(plaqueImage[rowstart:rowend,colstart:colend]>0)/plaqueSizeFactor
#             print((plaqueImage[rowstart:rowend,colstart:colend]>0).shape)
#             labels[idx]=labels[idx]/(rowend-rowstart)/(colend-colstart)
#             print((rowend-rowstart)*(colend-colstart))
            idx+=1
    naddedplaque=idx
    print('plaque'+str(idx+coord.shape[0]))
    labelsRes=np.concatenate((labelsRes,labels[:idx]))
    
    for r in range(rowSplits):
        for c in range(colSplits):
            rowstart=r*stride
            rowend=min((r+1)*stride+overlap,image.shape[0])
            colstart=c*stride
            colend=min((c+1)*stride+overlap,image.shape[1])
            if np.sum(plaqueBinary[rowstart:rowend,colstart:colend])>0:
                continue
            imagerc=image[rowstart:rowend,colstart:colend]
#             plaqueBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]=1
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
#             imagerc=np.pad(imagerc,((0,diamThresh-imagerc.shape[0]),(0,diamThresh-imagerc.shape[1])))
            res[idx,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[idx]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
            idx+=1
#             plaqueRes=np.concatenate((plaqueRes,imagerc.reshape((1,nchannels,imagerc.shape[0],imagerc.shape[1]))),axis=0)
#             coord=np.concatenate((coord,np.array([r*stride+diamThresh/2,c*stride+diamThresh/2]).reshape((1,2))),axis=0)
    plaqueRes=np.concatenate((plaqueRes,res[:idx]),axis=0)
    coord=np.concatenate((coord,coordNeg[:idx]),axis=0)
    labelsRes=np.concatenate((labelsRes,np.repeat(0,idx-naddedplaque)))
    print('no plaque'+str(idx-naddedplaque))
    
    res=None
    coorNeg=None
    plaqueCentroidBinary=None
    plaqueBinary=None
    resAll=plaqueRes
    coordAll=coord
            
    if not split:
        return resAll,labels,rowSplits,colSplits
    imTrain,imValTest,labelsTrain,labelsValTest,coordTrain,coordValTest=train_test_split(resAll,labelsRes,coordAll,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest,labelsVal,labelsTest,coordVal,coordTest=train_test_split(imValTest,labelsValTest,coordValTest,test_size=test/(val+test), random_state=seed, shuffle=True)
    if not returnPos:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest
    else:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest,coordTrain,coordVal,coordTest
    
def loadandsplitPlaque_overlap_regrs_cellCluster(clusterlabels,cellCoords,minPt,plaqueImageName,plaqueSizeFactor,areaThresh,coord,cutoffradius,samplename,imagedir,diamThresh,overlap,val,test,ifFlip=False,minCutoff=6,seed=3,split=True,imagename='pi_sum.tif',minmaxscale=True,nchannels=1,returnPos=False):
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images',imagename)
    image=mpimg.imread(imagepath,'tif').copy()
    print(image.shape)
    
    plaqueImagepath=os.path.join(imagedir,samplename,'trimmed_images',plaqueImageName)
    plaqueImage=mpimg.imread(plaqueImagepath,'tif')[:,:,0]
#     print(np.sum(plaqueImage[:,:,0]>0))
        
    if ifFlip:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    #plaque images
    coordAll=np.zeros_like(coord)
    imagetoClusterRes=np.zeros(coord.shape[0])
    labelsRes=np.zeros(coord.shape[0])
    plaqueRes=np.zeros((coord.shape[0],nchannels,diamThresh,diamThresh))
    plaqueBinary=np.zeros_like(image)
    radius=int(diamThresh/2)
    idx=0
    for c in range(coord.shape[0]):
        centroid=coord[c]
        if centroid[0]>=image.shape[0]:
            print('row '+str(centroid[0]))
        if centroid[1]>=image.shape[1]:
            print('col '+str(centroid[1]))
        rowstart=max(0,centroid[0]-radius)
        rowend=min(image.shape[0],centroid[0]+radius)
        colstart=max(0,centroid[1]-radius)
        colend=min(image.shape[1],centroid[1]+radius)
        
        #find corresponding cluster
        clusterIdxRow=np.logical_and(cellCoords[:,0]>=rowstart,cellCoords[:,0]<rowend)
        clusterIdxCol=np.logical_and(cellCoords[:,1]>=colstart,cellCoords[:,1]<colend)
        clusterRes=clusterlabels[np.logical_and(clusterIdxRow,clusterIdxCol)]
        if clusterRes.size==0:
            print('no cells')
            continue
        clusterResMode,modeCounts=stats.mode(clusterRes,axis=None)
        if modeCounts[0]/clusterRes.size<minPt:
            print('mode less than thresh')
            continue
        imagetoClusterRes[idx]=clusterResMode[0]
            
        imagerc=image[rowstart:rowend,colstart:colend]
#         print(imagerc.shape)
        if minmaxscale:
            imagercmin=np.min(imagerc)
            imagercmax=np.max(imagerc)
            if imagercmin==imagercmax:
                print('no cells')
                imagerc=np.zeros_like(imagerc)
            else:
                imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
        plaqueRes[idx,:,:(rowend-rowstart),:(colend-colstart)]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
        plaqueBinary[max(0,centroid[0]-cutoffradius):min(image.shape[0],centroid[0]+cutoffradius),max(0,centroid[1]-cutoffradius):min(image.shape[1],centroid[1]+cutoffradius)]=1
        labelsRes[idx]=np.sum(plaqueImage[rowstart:rowend,colstart:colend]>0)/plaqueSizeFactor
        coordAll[idx]=centroid
        idx+=1
#         print((plaqueImage[rowstart:rowend,colstart:colend]>0).shape)
#         print((rowend-rowstart)*(colend-colstart))
#         labelsRes[c]=labelsRes[c]/(rowend-rowstart)/(colend-colstart)
    plaqueRes=plaqueRes[:idx]
    labelsRes=labelsRes[:idx]
    imagetoClusterRes=imagetoClusterRes[:idx]
    coordAll=coordAll[:idx]
    plaqueCentroidBinary=np.copy(plaqueBinary)
    
    stride=diamThresh-overlap
    rowSplits=int(np.floor(image.shape[0]/stride)+((image.shape[0]%stride-overlap)>(minDist*minCutoff)))
    colSplits=int(np.floor(image.shape[1]/stride)+((image.shape[1]%stride-overlap)>(minDist*minCutoff)))
    res=np.zeros((rowSplits*colSplits,nchannels,diamThresh,diamThresh))
    coordNeg=np.zeros((rowSplits*colSplits,2))
    labels=np.zeros(rowSplits*colSplits)
    imagetoCluster=np.zeros(rowSplits*colSplits)
    #print(rowSplits*colSplits)
    
    idx=0
    for r in range(rowSplits):
        for c in range(colSplits):
            rowstart=r*stride
            rowend=min((r+1)*stride+overlap,image.shape[0])
            colstart=c*stride
            colend=min((c+1)*stride+overlap,image.shape[1])
            if np.sum(plaqueCentroidBinary[rowstart:rowend, colstart:colend])<areaThresh:
                continue
                
            #find corresponding cluster
            clusterIdxRow=np.logical_and(cellCoords[:,0]>=rowstart,cellCoords[:,0]<rowend)
            clusterIdxCol=np.logical_and(cellCoords[:,1]>=colstart,cellCoords[:,1]<colend)
            clusterRes=clusterlabels[np.logical_and(clusterIdxRow,clusterIdxCol)]
            if clusterRes.size==0:
                print('no cells')
                continue
            clusterResMode,modeCounts=stats.mode(clusterRes,axis=None)
            if modeCounts[0]/clusterRes.size<minPt:
                print('mode less than thresh')
                continue
            imagetoCluster[idx]=clusterResMode[0]
            
            imagerc=image[rowstart:rowend,colstart:colend]
#             print(imagerc.shape)
            plaqueBinary[rowstart:rowend,colstart:colend]=1
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
#             imagerc=np.pad(imagerc,((0,diamThresh-imagerc.shape[0]),(0,diamThresh-imagerc.shape[1])))
            res[idx,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[idx]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
            labels[idx] = np.sum(plaqueImage[rowstart:rowend,colstart:colend]>0)/plaqueSizeFactor
#             print((plaqueImage[rowstart:rowend,colstart:colend]>0).shape)
#             labels[idx]=labels[idx]/(rowend-rowstart)/(colend-colstart)
#             print((rowend-rowstart)*(colend-colstart))
            idx+=1
    naddedplaque=idx
    print('plaque'+str(idx+coordAll.shape[0]))
    labelsRes=np.concatenate((labelsRes,labels[:idx]))
    
    for r in range(rowSplits):
        for c in range(colSplits):
            rowstart=r*stride
            rowend=min((r+1)*stride+overlap,image.shape[0])
            colstart=c*stride
            colend=min((c+1)*stride+overlap,image.shape[1])
            if np.sum(plaqueBinary[rowstart:rowend,colstart:colend])>0:
                continue
            #find corresponding cluster
            clusterIdxRow=np.logical_and(cellCoords[:,0]>=rowstart,cellCoords[:,0]<rowend)
            clusterIdxCol=np.logical_and(cellCoords[:,1]>=colstart,cellCoords[:,1]<colend)
            clusterRes=clusterlabels[np.logical_and(clusterIdxRow,clusterIdxCol)]
            if clusterRes.size==0:
                print('no cells')
                continue
            clusterResMode,modeCounts=stats.mode(clusterRes,axis=None)
            if modeCounts[0]/clusterRes.size<minPt:
                print('mode less than thresh')
                continue
            imagetoCluster[idx]=clusterResMode[0]
            
            imagerc=image[rowstart:rowend,colstart:colend]
#             plaqueBinary[r*stride:min((r+1)*stride+overlap,image.shape[0]),c*stride:min((c+1)*stride+overlap,image.shape[1])]=1
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
#             imagerc=np.pad(imagerc,((0,diamThresh-imagerc.shape[0]),(0,diamThresh-imagerc.shape[1])))
            res[idx,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[idx]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
            idx+=1
#             plaqueRes=np.concatenate((plaqueRes,imagerc.reshape((1,nchannels,imagerc.shape[0],imagerc.shape[1]))),axis=0)
#             coord=np.concatenate((coord,np.array([r*stride+diamThresh/2,c*stride+diamThresh/2]).reshape((1,2))),axis=0)
    plaqueRes=np.concatenate((plaqueRes,res[:idx]),axis=0)
    coordAll=np.concatenate((coordAll,coordNeg[:idx]),axis=0)
    labelsRes=np.concatenate((labelsRes,np.repeat(0,idx-naddedplaque)))
    imagetoClusterRes=np.concatenate((imagetoClusterRes,imagetoCluster[:idx]))
    print('no plaque'+str(idx-naddedplaque))
    
    res=None
    coorNeg=None
    plaqueCentroidBinary=None
    plaqueBinary=None
    resAll=plaqueRes
            
    if not split:
        return resAll,labels,rowSplits,colSplits
    imTrain,imValTest,labelsTrain,labelsValTest,coordTrain,coordValTest,clusterTrain,clusterValTest = train_test_split(resAll,labelsRes,coordAll,imagetoClusterRes,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest,labelsVal,labelsTest,coordVal,coordTest,clusterVal,clusterTest = train_test_split(imValTest,labelsValTest,coordValTest,clusterValTest,test_size=test/(val+test), random_state=seed, shuffle=True)
    if not returnPos:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest
    else:
        return imTrain,imVal,imTest,labelsTrain,labelsVal,labelsTest,coordTrain,coordVal,coordTest,clusterTrain,clusterVal,clusterTest
    
def loadandsplit_cellCluster(clusterlabels,cellCoords,minPt,samplename,imagedir,diamThresh,overlap,val,test,ifFlip=True,minCutoff=6,seed=3,split=True,imagename='pi_sum.tif',minmaxscale=True,nchannels=1,clf=False,returnPos=False):
    ###under construction
#     diamThresh=minDist*diamThresh_mul
    imagepath=os.path.join(imagedir,samplename,'trimmed_images',imagename)
    image=mpimg.imread(imagepath,'tif').copy()
    print(image.shape)
    if len(image.shape)==3:
        image=image[:,:,0]
    
    if ifFlip:
        print('flipping images')
        if samplename=='AD_mouse9494':
            image=np.flipud(image)
        elif samplename=='AD_mouse9498':
            image=np.fliplr(image)
        elif samplename=='AD_mouse9735':
            image=np.fliplr(image)
            image=np.flipud(image)
    
    stride=diamThresh-overlap
    rowSplits=int(np.floor(image.shape[0]/stride)+((image.shape[0]%stride-overlap)>(minDist*minCutoff)))
    colSplits=int(np.floor(image.shape[1]/stride)+((image.shape[1]%stride-overlap)>(minDist*minCutoff)))
    res=np.zeros((rowSplits*colSplits,nchannels,diamThresh,diamThresh))
    coordNeg=np.zeros((rowSplits*colSplits,2))
    imagetoCluster=np.zeros(rowSplits*colSplits)
    idx=0
    
    for r in range(rowSplits):
        for c in range(colSplits):
            rowstart=r*stride
            rowend=min((r+1)*stride+overlap,image.shape[0])
            colstart=c*stride
            colend=min((c+1)*stride+overlap,image.shape[1])
            
            #find corresponding cluster
            clusterIdxRow=np.logical_and(cellCoords[:,0]>=rowstart,cellCoords[:,0]<rowend)
            clusterIdxCol=np.logical_and(cellCoords[:,1]>=colstart,cellCoords[:,1]<colend)
            clusterRes=clusterlabels[np.logical_and(clusterIdxRow,clusterIdxCol)]
            if clusterRes.size==0:
                print('no cells')
                continue
            clusterResMode,modeCounts=stats.mode(clusterRes,axis=None)
            if modeCounts[0]/clusterRes.size<minPt:
                print('mode less than thresh')
                continue
            imagetoCluster[idx]=clusterResMode[0]
           
            imagerc=image[rowstart:rowend,colstart:colend]
            if minmaxscale:
                imagercmin=np.min(imagerc)
                imagercmax=np.max(imagerc)
                if imagercmin==imagercmax:
                    print('no cells')
                    imagerc=np.zeros_like(imagerc)
                else:
                    imagerc=(imagerc-imagercmin)/(imagercmax-imagercmin)
            res[idx,:,:imagerc.shape[0],:imagerc.shape[1]]=imagerc.reshape((nchannels,imagerc.shape[0],imagerc.shape[1]))
            coordNeg[idx]=np.array([r*stride+diamThresh/2,c*stride+diamThresh/2])
            idx+=1
    res=res[:idx]
    coordNeg=coordNeg[:idx]
    imagetoCluster=imagetoCluster[:idx]
            
#     image=None
    if not split:
        if not returnPos:
            return res,rowSplits,colSplits
        else:
            return res,coordNeg,rowSplits,colSplits
    imTrain,imValTest,coordTrain,coordNegValTest,clusterTrain,clusterValTest = train_test_split(res,coordNeg,imagetoCluster,test_size=val+test, random_state=seed, shuffle=True)
    imVal,imTest,coordVal,coordTest,clusterVal,clusterTest = train_test_split(imValTest,coordNegValTest,clusterValTest,test_size=test/(val+test), random_state=seed, shuffle=True)
    if clf:
        if not returnPos:
            return imTrain,imVal,imTest,np.zeros(imTrain.shape[0]),np.zeros(imVal.shape[0]),np.zeros(imTest.shape[0])
        else:
            return imTrain,imVal,imTest,np.zeros(imTrain.shape[0]),np.zeros(imVal.shape[0]),np.zeros(imTest.shape[0]), coordTrain,coordVal,coordTest,clusterTrain,clusterVal,clusterTest
    else:
        return imTrain,imVal,imTest