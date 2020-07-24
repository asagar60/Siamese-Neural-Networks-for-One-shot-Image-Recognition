# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:07:52 2020

@author: asaga
"""

import os
import random
import numpy as np 
import glob
import pickle as pkl
import h5py
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from tqdm import tqdm
from time import time
from PIL import Image

class dataloader:
    
    def __init__(self,train_path = 'images_background', test_path = 'images_evaluation'):
        
        self.train_path = train_path
        self.test_path = test_path
        
        
    def saveImagePaths(self,  setType = 'Train' ):
        
        folder_path = self.train_path
        image_paths = []
        for i,_dir in enumerate(os.listdir(folder_path)):
            _dirpath = os.path.join(folder_path, _dir)
            dir_images = []
            for _subdir in os.listdir(_dirpath):
                _subdirpath = os.path.join(_dirpath, _subdir)
                img_path_list =  glob.glob(_subdirpath+"/*.png")
                dir_images.append(img_path_list)
            
            dir_images = np.array(dir_images)
            #dir_images = np.reshape(dir_images, (1,dir_images.shape[0], dir_images.shape[1]))
                
            image_paths.append(dir_images)
        image_paths = np.asarray(image_paths)
        
        self.file_name = '{}_im_paths.pkl'.format(setType)
        
        with open(self.file_name, "wb") as f:
            pkl.dump(image_paths, f)
            print()
            print(self.file_name+" saved")
        
    def generateTrainingPairs(self, n = 183160):
    
        pairs,y = [],[]
        with open(self.file_name, 'rb') as f:
            image_paths = pkl.load(f)
        
        train_filename ='training_file_{}.pkl'.format(n)
        
        total = 0
        for i in tqdm(range(len(image_paths))):
            for j in range(len(image_paths[i])):
                for k in range(len(image_paths[i][j])):
                    if total == 2*n:
                        X, y = np.array(pairs), np.array(y)
                        with open(train_filename, "wb") as f:
                            pkl.dump((X,y), f)
                            print()
                            print(train_filename+" saved")
                        return
                        
                    path_1 = image_paths[i][j][k]
                    
                    for m in range(k+1,len(image_paths[i][j])):
                        path_2 = image_paths[i][j][m]
                        pairs.append([path_1, path_2])
                        y.append(1)
                        
                        path_2 = random.sample(list(random.sample(list(random.sample(list(image_paths),1)[0]),1)[0]),1)[0]
                        pairs.append([path_1, path_2])
                        y.append(0)
                        
                        total = total+2
            
    def val_eval_split(self):
    
        folder_path = self.test_path    
    
        eval_list = ['Atlantean', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Malayalam', 'Manipuri', 'Old_Church_Slavonic_(Cyrillic)' ,'Tengwar','Tibetan']
        dir_list = os.listdir(folder_path)
        val_dir = [dir_ for dir_ in dir_list if dir_ not in eval_list]
        
        #Type 1 - validation data , evaluation data
        self.wA_test_pairs(folder_path, val_dir, savefilename = 'wA_val_10_split_images.pkl', n_way = 20)
        self.uA_test_pairs(folder_path, val_dir, savefilename = 'uA_val_10_split_images.pkl',  n_way = 20)
        
        self.wA_test_pairs(folder_path, eval_list, savefilename = 'wA_eval_10_split_images.pkl',  n_way = 20)
        self.uA_test_pairs(folder_path,eval_list, savefilename = 'uA_eval_10_split_images.pkl',  n_way = 20)
        
        #Type 2 - Validation + evaluation
        
        self.wA_test_pairs(folder_path,eval_list, savefilename = 'wA_eval_20_split_images.pkl',  n_way = 20)
        self.uA_test_pairs(folder_path, eval_list, savefilename = 'uA_eval_20_split_images.pkl',  n_way = 20)
        

    def wA_test_pairs(self, folder_path, dirs, savefilename, n_way = 20):
        X,y = [],[]
        for alpha in dirs:
            alphabet_dir = os.path.join(folder_path,alpha)
            char_dirs = os.listdir(alphabet_dir)
            char_dirs = random.sample(char_dirs,n_way)
            set_1, set_2 = [],[]
            for char in char_dirs:
                char_path = os.path.join(alphabet_dir, char)
                img_paths =  glob.glob(char_path+"/*.png")
                random_samples = random.sample(img_paths,2)
                set_1.append(random_samples[0])
                set_2.append(random_samples[1])

            for i,imPath1 in enumerate(set_1):
                for j,imPath2 in enumerate(set_2):
                    img1 = np.expand_dims(mpimg.imread(imPath1), axis = 2)
                    img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                    X.append([img1, img2])
                    y.append(1 if i==j else 0)
            
            for i,imPath1 in enumerate(set_2):
                for j,imPath2 in enumerate(set_1):
                    img1 = np.expand_dims(mpimg.imread(imPath1), axis = 2)
                    img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                    X.append([img1, img2])
                    y.append(1 if i==j else 0)
        X, y = np.array(X), np.array(y)
        #y = np.reshape(y,(-1,1))

        if savefilename == None:
            return X,y
        else:
            
            with open(savefilename, "wb") as f:
                pkl.dump((X,y), f)
                print()
                print(savefilename+" saved")
            
    
    def uA_test_pairs(self,folder_path, dirs, savefilename, classes = None, n_way = 20):
            
        X,y = [],[]
        
        if classes == None:
            dirs = random.sample(dirs, len(dirs))
        else:
            dirs = random.sample(dirs, classes)
            
        for alpha in dirs:
            alphabet_dir = os.path.join(folder_path,alpha)
            char_dirs = os.listdir(alphabet_dir)
            char_dirs = random.sample(char_dirs,n_way)
            for char in char_dirs:
                
                char_path = os.path.join(alphabet_dir, char)
                img_paths =  glob.glob(char_path+"/*.png")
                
                
                imPath1, imPath2 = random.sample(img_paths,2)
                img1 = np.expand_dims(mpimg.imread(imPath1), axis = 2)
                img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                X.append([img1, img2])
                y.append(1)
                
                for _ in range(n_way-1):
                    random_alpha_pick = random.sample(dirs,1)[0]
                    random_alphabet_dir = os.path.join(folder_path,random_alpha_pick)
                    random_char_dirs = os.listdir(random_alphabet_dir)
                    random_pick = random.sample(random_char_dirs,1)[0]
                    
                    
                    while(random_pick == char):
                        random_pick = random.sample(random_char_dirs,1)[0]
                    
                    
                    random_char_dir = os.path.join(random_alphabet_dir,random_pick)
                    imPath2 = random.sample(glob.glob(random_char_dir+"/*.png"),1)[0]
                    img2 = np.expand_dims(mpimg.imread(imPath2), axis = 2)
                    X.append([img1, img2])
                    y.append(0)
        X, y = np.array(X), np.array(y)
        #y = np.reshape(y,(-1,1))
               
        if savefilename == None:
            return (X,y)
        else:
            
            with open(savefilename, "wb") as f:
                pkl.dump((X,y), f)
                print()
                print(savefilename+" saved")
            
            
            
            
            
if __name__=='__main__':
    data = dataloader(train_path = 'images_background', test_path = 'images_evaluation')
    data.saveImagePaths()
    data.generateTrainingPairs()
    data.val_eval_split()