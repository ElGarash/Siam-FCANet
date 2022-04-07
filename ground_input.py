import argparse
import pickle
import torch
from torch.autograd import Variable

from torchvision import  transforms
from torch.utils.data import Dataset, DataLoader

from SiamFCANet import SiamFCANet18_CVUSA

from input_data import InputData

import numpy as np
import os

from PIL import Image 

from numpy.random import randint as randint
from numpy.random import uniform as uniform


########################
torch.backends.cudnn.benchmark = True # use cudnn
########################

class ImageDataForExam(Dataset):
    def __init__(self, grd_list): 
            
        self.image_names_grd = grd_list
    
    def __len__(self):

        return len(self.image_names_grd)

    def __getitem__(self, idx):

        ### for query data
        data_names_grd = os.path.join('', self.image_names_grd[idx])
        image_grd = Image.open(data_names_grd)
        
        ### adjust with torchvision
        trans_img_G = transforms.ToTensor()(image_grd)
        # torchvision is R G B and opencv is B G R
        trans_img_G[0] = trans_img_G[0]*255.0 - 123.6  # Red
        trans_img_G[1] = trans_img_G[1]*255.0 - 116.779  # Green
        trans_img_G[2] = trans_img_G[2]*255.0 - 103.939  # Blue
        
        return trans_img_G

#################
DESCRIPTORS_DIRECTORY = '/kaggle/working/descriptors/SFCANet18'
# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="The path of the ground image to calculate the distance for")
args = parser.parse_args()
print(args)
    
#######################

mini_batch = 8

########################### Feature Extraction ############################
### feature vectors generation
def FeatVecGen(net_test):
    ### net evaluation state
    net_test.eval()
    
    filenames_query = [args.image_path]
    
    my_data = ImageDataForExam(filenames_query)
                                     
    mini_batch = 8
    testloader = DataLoader(my_data, batch_size=mini_batch, shuffle=False, num_workers=8)
    
    N_data = len(filenames_query)
    vec_len = 1024
    
    ### N_data % mini_batch 
    nail = N_data % mini_batch
    ### N_data // mini_batch 
    max_i = N_data // mini_batch
    ### creat a space for restoring features
    query_vec = np.zeros([N_data,vec_len], dtype=np.float32)
    
    ### feature extraction
    for i, data in enumerate(testloader, 0):
        data_query = data
        data_query = Variable(data_query).cuda()
        
        outputs_query, _ = net_test.forward_SV(data_query)
        
        ###### feature vectors feeding
        if(i<max_i):
            m = mini_batch*i
            n = mini_batch*(i+1)
            query_vec[m:n] = outputs_query.data.cpu().numpy()
        else:
            m = mini_batch*i
            n = mini_batch*i + nail
            query_vec[m:n] = outputs_query.data.cpu().numpy()
        

    
    with open(f"{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl", 'rb') as f:
        sat_global_descriptor = pickle.load(f)

    return query_vec, sat_global_descriptor



##########################

### Siam-FCANet 18 ###


net = SiamFCANet18_CVUSA()
net.cuda()

weight_path = '/kaggle/working/'
net.load_state_dict(torch.load(weight_path+'SFCANet_18.pth'))

ground_descriptors, satellite_descriptors = FeatVecGen(net)


### vectors import
N_data = ground_descriptors.shape[0]
N_data_sat = satellite_descriptors.shape[0]
###

dist_E = (satellite_descriptors**2).sum(1).reshape(N_data_sat,1)
dist_Q = (ground_descriptors**2).sum(1).reshape(N_data,1).T

dist_array = dist_E + dist_Q - 2 * np.matmul(satellite_descriptors, ground_descriptors.T)

print('dist_array shape', dist_array.shape)

with open(f"{DESCRIPTORS_DIRECTORY}/dist_array.pkl", 'wb') as f:
    pickle.dump(dist_array, f)
