import os
import yaml
import torch
import monai
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot
sitk.ProcessObject.SetGlobalWarningDisplay(False)
from typing import Tuple, List, Mapping, Hashable, Dict
from monai.transforms import (
    LoadImaged, MapTransform, ScaleIntensityRanged, EnsureChannelFirstd, Spacingd, Orientationd,ResampleToMatchd, ResizeWithPadOrCropd, Resize, Resized, RandFlipd, NormalizeIntensityd, ToTensord,RandScaleIntensityd,RandShiftIntensityd
)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(self, keys: monai.config.KeysCollection,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: monai.config.NdarrayOrTensor):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [(img == 1)|(img == 3)]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.concatenate(result, axis=0).astype(np.float32)

    def __call__(self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]) -> Dict[
        Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
    
def load_MR_dataset_images(root, usedata, use_models):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        if path in usedata:
            models = os.listdir(root + '/' + path + '/')
            image = []
            label = []
            for model in models:
                if model in use_models:
                    image.append(root + '/' + path + '/' + model + '/' + path + '.nii.gz')
                    label.append(root + '/' + path + '/' + model + '/' + path + 'seg.nii.gz')
            images_list.append({
                'image': image,
                'label': label
            })
            
    return images_list

def read_usedata(file_path):
    read_flas = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if read_flas == True:
                result = line.replace('\n', '').split(',')
                result = [data.replace(' ', '') for data in result]
                return result
            elif 'Useful data' in line:
                read_flas = True
                continue

def get_transforms(config: EasyDict) -> Tuple[
    monai.transforms.Compose, monai.transforms.Compose]:
    
    load_transform = []
    
    for model_scale in config.loader.model_scale:
        load_transform.append(
            monai.transforms.Compose([
                LoadImaged(keys=["image", "label"], image_only=False, simple_keys=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=config.loader.target_size, mode=("trilinear", "nearest-exact")),
                
                ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,            # 输出图像的最小强度值
                        b_max=1.0,            # 输出图像的最大强度值
                        clip=True             # 是否裁剪超出范围的值
                    ),
                ToTensord(keys=['image', 'label'])
            ])
        )
    
    train_transform = monai.transforms.Compose([
        # 训练集的额外增强
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ])
    val_transform = monai.transforms.Compose([
        ToTensord(keys=["image", "label"]),
    ])
    return load_transform, train_transform, val_transform

class MultiModalityDataset(monai.data.Dataset):
    def __init__(self, data, loadforms, transforms):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        combined_data = {}
        
        for i in range(0, len(item['image'])):
            globals()[f'data_{i}'] = self.loadforms[i]({
                'image': item['image'][i],
                'label': item['label'][i]
            })

            combined_data[f'model_{i}_image'] = globals()[f'data_{i}']['image']
            combined_data[f'model_{i}_label'] = globals()[f'data_{i}']['label']
        
        images = []
        labels = []
        
        for i in range(0, len(item['image'])):
            images.append(combined_data[f'model_{i}_image'])
            labels.append(combined_data[f'model_{i}_label'])
            image_tensor = torch.cat(images, dim=0)
            label_tensor = torch.cat(labels, dim=0)
        
        result = {'image': image_tensor, 'label': label_tensor}
        result = self.transforms(result)
        return {'image': image_tensor, 'label': label_tensor}

def get_dataloader(config: EasyDict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    datapath = config.loader.dataPath
    use_models = config.loader.checkModels
    # TODO: 统一数据名称
    datapath1 = datapath + '/' + 'NonsurgicalMR' + '/'
    datapath2 = datapath + '/' + 'SurgicalMR' + '/'
    usedata1 = datapath + '/' + 'NonsurgicalMR.txt'
    usedata2 = datapath + '/' + 'SurgicalMR.txt'
    usedata1 = read_usedata(usedata1)
    usedata2 = read_usedata(usedata2)
    
    data1 = load_MR_dataset_images(datapath1, usedata1, use_models)
    data2 = load_MR_dataset_images(datapath2, usedata2, use_models)
    
    data = data1 + data2
    
    load_transform, train_transform, val_transform = get_transforms(config)
    
    train_data = data[:int(len(data) * config.loader.train_ratio)]
    val_data   = data[int(len(data) * config.loader.train_ratio):]
    
    train_dataset = MultiModalityDataset(data=train_data, 
                                         loadforms = load_transform,
                                         transforms=train_transform)
    val_dataset   = MultiModalityDataset(data=val_data, 
                                         loadforms = load_transform,
                                         transforms=val_transform)
    
    train_loader = monai.data.DataLoader(train_dataset, num_workers=config.loader.num_workers,
                                         batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = monai.data.DataLoader(val_dataset, num_workers=config.loader.num_workers, 
                                       batch_size=config.trainer.batch_size, shuffle=False)
    return train_loader, val_loader


def print_unique_elements(tensor):
    """
    打印给定张量中所有不同的元素。
    
    参数:
        tensor (torch.Tensor): 要检查不同元素的张量。
    """
    # 获取不同元素
    unique_elements = torch.unique(tensor)
    
    print("Unique elements in the tensor:")
    for element in unique_elements:
        print(element.item(), end=' ')  # 使用 .item() 来获取 Python 标量值，并在同一行打印
    return unique_elements
    
if __name__ == '__main__':

    config = EasyDict(yaml.load(open('/workspace/Jeming/MR/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    train_loader, val_loader = get_dataloader(config)
    
    count = 0
    f_max = 0
    s_max = 0 
    for i, batch in enumerate(train_loader):
        # print(batch['image'].shape)
        # print(batch['label'].shape)
        if batch['image'][0][0].max().item() > f_max:
            f_max = batch['image'][0][0].max().item()
        if batch['image'][0][1].max().item() > s_max:
            s_max = batch['image'][0][1].max().item()
        print('now f max: ', f_max)
        print('now s max: ', s_max)
        print('\n')
        count += 1
        
    count = 0
    for i, batch in enumerate(val_loader):
        # print(batch['image'].shape)
        # print(batch['label'].shape)
        if batch['image'][0][0].max().item() > f_max:
            f_max = batch['image'][0][0].max().item()
        if batch['image'][0][1].max().item() > s_max:
            s_max = batch['image'][0][1].max().item()
        print('now f max: ', f_max)
        print('now s max: ', s_max)
        print('\n')
        count += 1