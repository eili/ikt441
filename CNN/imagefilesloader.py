from os import listdir
from os.path import isfile, join, expanduser
from PIL import Image
import numpy as np


class Imagefilesloader:

    def __init__(self):
        self.filecount = 0
        self.sizex = 100
        self.sizey = 100
        self.train_labels = []
        self.genderlist = []

     #
    def getallfilenames(self, imagefolder):
        result = dict()
        self.filecount = 0
        personfolders = listdir(imagefolder)
        for personfolder in personfolders:
            if personfolder.startswith("."):
                continue
            imagegrouppath = join(imagefolder, personfolder)
            print(personfolder)
            files = [f for f in listdir(imagegrouppath) if isfile(join(imagegrouppath, f))]
            personfiles = []
            for f in files:
                if f.startswith("."):
                    continue
                imagefile = join(imagegrouppath, f)
                personfiles.append(imagefile)
                self.filecount += 1
            result[personfolder] = personfiles

        return result

    def getdata(self, imagefolder):
        #imagefolder = "./lfw/"
        # result is a dictionary with the persons name as the key, and an array of images as the value.
        # the images are flattened to a tuple of three sets of ints, for RGB.
        # https://github.com/teammcr192/deep-CNN-keras/blob/master/img_loader.py
        allfiles = self.getallfilenames(imagefolder)

        personFolders = listdir(imagefolder)
        train_data = np.empty((self.filecount, 3, self.sizex, self.sizey),
                                   dtype='float32')

        i = 0
        image_index = 0
        for key, val in list(allfiles.items()):
            for filename in val:
                img = Image.open(filename, 'r')
                img = img.resize((self.sizex, self.sizey))
                img_arr = np.asarray(img, dtype='float32')
                train_data[image_index, 0, :, :] = img_arr[:, :, 0]
                train_data[image_index, 1, :, :] = img_arr[:, :, 1]
                train_data[image_index, 2, :, :] = img_arr[:, :, 2]
                self.train_labels.append(key)
                image_index += 1

        train_data = train_data.astype('float32') / 255
        return train_data


    def getdataForFileArray(self, count, allfiles):
        train_data = np.empty((count, 3, self.sizex, self.sizey), dtype='float32')
        self.genderlist = []

        image_index = 0
        for key, val in list(allfiles.items()):
            for filename in val:
                img = Image.open(filename, 'r')
                img = img.resize((self.sizex, self.sizey))

                img_arr = np.asarray(img, dtype='float32')
                train_data[image_index, 0, :, :] = img_arr[:, :, 0]
                train_data[image_index, 1, :, :] = img_arr[:, :, 1]
                train_data[image_index, 2, :, :] = img_arr[:, :, 2]
                #self.train_labels.append(key)
                image_index += 1

        train_data = train_data.astype('float32') / 255
        return train_data


    def getgenderata(self, basepath):
        foldermale = join(basepath, "male/")
        folderfemale = join(basepath, "female/")
        malefiles = self.getallfilenames(foldermale)
        malefilecount = self.filecount
        femalefiles = self.getallfilenames(folderfemale)
        femalefilecount = self.filecount

        print("Males", malefilecount)
        print("Females", femalefilecount)


        maleset = self.getdataForFileArray(malefilecount, malefiles)
        femaleset = self.getdataForFileArray(femalefilecount, femalefiles)
        l1 = np.zeros(malefilecount)
        l2 = np.ones(femalefilecount)
        ylist = np.concatenate([l1, l2])

        totalset = np.concatenate([maleset, femaleset])
        return (totalset, ylist)

