from torch.utils.data import Dataset
import numpy as np

def get_lc_split():
    train_file_names = np.load('./train.npy', allow_pickle=True).tolist()
    val_file_names = np.load('./val.npy', allow_pickle=True).tolist()
    test_file_names = np.load('./test.npy', allow_pickle=True).tolist()
    test_video = {'test_frame': test_file_names}
    return train_file_names, val_file_names, test_file_names, test_video



class LC_Dataset(Dataset):
    def __init__(self, file_names, transforms, phase):
        self.file_names = file_names
        self.transforms = transforms
        self.data_path = phase + '_data.npy'
        self.data = np.load(self.data_path, allow_pickle=True).item()


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]

        image = self.data[str(img_file_name)]
        mask_file_name = str(img_file_name).replace('image', 'mask')
        identifier = mask_file_name.split('/')[-1]
        mask_file_name = mask_file_name.replace(identifier, identifier[:-4] + '_watershed_mask' + '.png')
        mask = self.data[mask_file_name]
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        label = transformed["mask"].long()

        sample = {'image': image, 'label': label, 'id': str(img_file_name).split('/')[-1]}
        return sample
