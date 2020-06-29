# -*- coding: utf-8 -*-
import os
imgpath = "/home/miku/imagedata/VisDrone2019-DET-train/images"
labelpath = "/home/miku/imagedata/VisDrone2019-DET-train/annotations"
filelist = os.listdir(imgpath) #该文件夹下所有的文件（包括文件夹）
filelab = os.listdir(labelpath) #该文件夹下所有的文件（包括文件夹）
count=1
imagetype = os.path.splitext(filelist[0])[1]
labeltype = os.path.splitext(filelab[0])[1]
for file in filelist:   #遍历所有文件
    filename = os.path.splitext(file)[0]
    Oldimgpath = os.path.join(imgpath+'/', filename+imagetype )   #原来的文件路径
    Oldlabpath = os.path.join(labelpath+'/', filename+labeltype)  # 原来的文件路径
    if os.path.isdir(Oldimgpath):   #如果是文件夹则跳过
        continue
    if os.path.isdir(Oldlabpath):   #如果是文件夹则跳过
        continue
    Newimgpath = os.path.join(imgpath+'/', str(count).zfill(6) + imagetype)  #用字符串函数zfill 以0补全所需位数
    Newlabpath = os.path.join(labelpath + '/', str(count).zfill(6) + labeltype)  # 用字符串函数zfill 以0补全所需位数
    os.rename(Oldimgpath, Newimgpath)#重命名
    os.rename(Oldlabpath, Newlabpath)  #重命名
    count += 1