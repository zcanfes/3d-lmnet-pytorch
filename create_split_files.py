#Shapenet pointcloud directory is located in the below path
path=""
# train splits will be saved
train_path=""
#test splits will be saved
test_path=""
#validation splits will be saved
val_path=""
import os
import numpy as np
train=[]
test=[]
val=[]
for _, dirs, _  in os.walk(path):
	#print(dirs)
	for dir in dirs:
		#print(dir)
		#for variational part of the model, only chair class is required. So uncomment the cells below only to split the chair class into train, val and test
		"""
		if dir!="03001627":
			continue
		"""
		#when only 3 different classes (airplane, car, chair) are wanted to use, then uncomment the cell below
		"""
		if dir!="02691156" and dir!="02958343" and dir!="03001627":
			continue
		"""

		for _, cats, _ in os.walk(path+"/"+dir):
			c_files=[]
			#print(cats[0])
			
			for cat in cats:
				c_files.append(dir+"/"+cat)
			idx=np.random.permutation(len(c_files))

			tr_idx=round(len(c_files)*0.6)
			v_idx=round(len(c_files)*0.2)
			c_files_= np.asarray(c_files, dtype=object)

			tr,v,ts=c_files_[idx[:tr_idx]],c_files_[idx[tr_idx:tr_idx+v_idx]],c_files_[idx[tr_idx+v_idx:]]

			train.extend(tr)
			val.extend(v)
			test.extend(ts)

with open(train_path, 'w') as the_file:
    for el in train:
        the_file.write(el+'\n')

with open(val_path, 'w') as the_file:
    for el in val:
        the_file.write(el+'\n')

with open(test_path, 'w') as the_file:
    for el in test:
        the_file.write(el+'\n')
