import os, shutil, random, const
import seedir as sd
from tqdm import tqdm


def make_dir(dir_path):
    # If a folder already exists on the path address: (str) dir_path,
    if os.path.exists(dir_path):
        # first delete the existing folder,
        shutil.rmtree(dir_path)
    # then create a new folder.
    os.makedirs(dir_path)


Dataset_Path = const.LABELME_DATASET_PATH
Images_Path = os.path.join(Dataset_Path, const.IMAGES)
Images_List = os.listdir(Images_Path)

random.seed(1)
random.shuffle(Images_List)

Test_Rate = const.TEST_RATE

Val_Number   = int(len(Images_List) * Test_Rate)
Train_Images = Images_List[Val_Number:]
Val_Images   = Images_List[:Val_Number]

print(f'Total number of dataset files: {len(Images_List)}.')
print('Total number of dataset files for training: ', len(Train_Images))
print('Total number of dataset files for validation: ', len(Val_Images))

Train_Path_Images = os.path.join(const.OUTPUT_PATH, const.IMAGES)
Train_Path_Images = os.path.join(Train_Path_Images, 'train')
make_dir(Train_Path_Images)
for each in tqdm(Train_Images):
    each_path = os.path.join(Images_Path, each)
    shutil.copy(each_path, Train_Path_Images)

Val_Path_Images = os.path.join(const.OUTPUT_PATH, const.IMAGES)
Val_Path_Images = os.path.join(Val_Path_Images, 'val')
make_dir(Val_Path_Images)
for each in tqdm(Val_Images):
    each_path = os.path.join(Images_Path, each)
    shutil.copy(each_path, Val_Path_Images)
# ---------------------------------------------------------------------------------------------------
Labels_Path = os.path.join(Dataset_Path, const.LABELME)

Train_Path_Labels = os.path.join(const.OUTPUT_PATH, const.LABELS)
Train_Path_Labels = os.path.join(Train_Path_Labels, 'train')
make_dir(Train_Path_Labels)
for each in tqdm(Train_Images):
    each = each.split('.')[0] + '.json'
    each_path = os.path.join(Labels_Path, each)
    shutil.copy(each_path, Train_Path_Labels)

Val_Path_Labels = os.path.join(const.OUTPUT_PATH, const.LABELS)
Val_Path_Labels = os.path.join(Val_Path_Labels, 'val')
make_dir(Val_Path_Labels)
for each in tqdm(Val_Images):
    each = each.split('.')[0] + '.json'
    each_path = os.path.join(Labels_Path, each)
    shutil.copy(each_path, Val_Path_Labels)
# ---------------------------------------------------------------------------------------------------
sd.seedir(const.OUTPUT_PATH, style='emoji', depthlimit=2)
