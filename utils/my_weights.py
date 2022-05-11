import torch
import os
import glob

my_folder = os.path.join('./', 'saved', 'models', 'my_model', '0125_145644')
for file in os.listdir(my_folder):
     # check the files which are end with specific extension
    if file.endswith('.pth'):
        if file.startswith('checkpoint'):
        # print path name of selected files
            print(os.path.join(my_folder, file))
            checkpoint = torch.load(os.path.join(my_folder, file))
            state_dict = checkpoint['state_dict']
            print('done.')