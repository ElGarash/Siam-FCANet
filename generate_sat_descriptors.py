import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from SiamFCANet import SiamFCANet18_CVUSA
from input_data import InputData
import numpy as np
import os
from PIL import Image 
from numpy.random import randint as randint
from numpy.random import uniform as uniform
import pickle


########################
torch.backends.cudnn.benchmark = True # use cudnn
########################

##### for testing #####

class ImageDataForExam(Dataset):
    ###label_list 0 1  A means Anchor and P means positive
    def __init__(self, sat_list): 
        
        self.image_names_sat = sat_list
        
        ######
    
    def __len__(self):

        return len(self.image_names_sat)

    def __getitem__(self, idx):
        ###### for examing data
        data_names_sat = os.path.join('', self.image_names_sat[idx])
        image_sat = Image.open(data_names_sat)
        
        ### adjust with torchvisison
        trans_img_S = transforms.Resize([512,512], interpolation=Image.ANTIALIAS)(image_sat)
        trans_img_S = transforms.ToTensor()(trans_img_S)
        
        trans_img_S[0] = trans_img_S[0]*255.0 - 123.6  # Red
        trans_img_S[1] = trans_img_S[1]*255.0 - 116.779  # Green
        trans_img_S[2] = trans_img_S[2]*255.0 - 103.939  # Blue
        
        ########################################
        
        return trans_img_S

#################

### load data
data = InputData()
testList = data.id_test_list
#######################
up_root = '/kaggle/input/cvusa-dataset/cvusa-localization/'


### vectors restoring path
save_path = '/kaggle/working/'

###########################

mini_batch = 8

########################### Feature Extraction ############################
### feature vectors generation
def FeatVecGen(net_test, model_name):
    ### net evaluation state

    net_test.eval()
    filenames_examing = []

    for sample in testList:
        info_examing = up_root + sample[0]
        filenames_examing.append(info_examing)
    
    my_data = ImageDataForExam(filenames_examing)
                                     
    mini_batch = 8
    testloader = DataLoader(my_data, batch_size=mini_batch, shuffle=False, num_workers=8)
    
    N_data = len(filenames_examing)
    vec_len = 1024
    
    ### N_data % mini_batch 
    nail = N_data % mini_batch
    ### N_data // mini_batch 
    max_i = N_data // mini_batch
    ### creat a space for restoring features
    examing_vec = np.zeros([N_data,vec_len], dtype=np.float32)
    
    ### feature extraction
    for i, data in enumerate(testloader, 0):
        data_examing = data
        data_examing = Variable(data_examing).cuda()
        
        outputs_examing, _ = net_test.forward_OH(data_examing)
        
        ###### feature vectors feeding
        if(i<max_i):
            m = mini_batch*i
            n = mini_batch*(i+1)
            examing_vec[m:n] = outputs_examing.data.cpu().numpy()
        else:
            m = mini_batch*i
            n = mini_batch*i + nail
            examing_vec[m:n] = outputs_examing.data.cpu().numpy()
        
        if(i % 8 == 0):
            print(i)
    
    DESCRIPTORS_DIRECTORY = '/kaggle/working/descriptors/SFCANet18'

    if not os.path.exists(DESCRIPTORS_DIRECTORY):
            os.makedirs(DESCRIPTORS_DIRECTORY)


    with open(f"{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl", 'wb') as f:
        pickle.dump(examing_vec, f)
        
    print('Satellite images descriptors successfully generated')



##########################

### Siam-FCANet 18 ###

model_name = 'SFCANet18'

net = SiamFCANet18_CVUSA()
net.cuda()

weight_path = '/kaggle/working/'
net.load_state_dict(torch.load(weight_path+'SFCANet_18.pth'))

FeatVecGen(net, model_name)