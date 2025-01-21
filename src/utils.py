import os
import random
import sys
from collections import OrderedDict

import monai
import math
import numpy as np
import torch
from accelerate import Accelerator
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from torch import nn
from pathlib import Path
import numpy as np
import nibabel as nib

class MetricSaver(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_acc = nn.Parameter(torch.zeros(1), requires_grad=False)


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()


def ensure_directory_exists(directory_path):
    path = Path(directory_path)
    
    # 如果路径存在并且是一个目录，则直接返回
    if path.exists() and path.is_dir():
        return
    
    # 创建目录及其所有必要的父级目录
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"create directory successly: {directory_path}")
    except Exception as e:
        print(f"create directory failed: {directory_path}, message: {e}")

def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(download_path, map_location=torch.device('cpu'))
    return state_dict


def resume_train_state(model, checkpoint, optimizer, scheduler, accelerator):
    try:
        base_path = f"{os.getcwd()}/model_store/{checkpoint}/"
        epoch_checkpoint = torch.load(base_path + "/epoch.pth.tar", map_location=accelerator.device)
        best_score = epoch_checkpoint['best_score']
        best_metrics = epoch_checkpoint['best_metrics']
        starting_epoch = epoch_checkpoint['epoch'] + 1
        train_step = epoch_checkpoint['train_step']
        val_step = epoch_checkpoint['val_step']
        model = load_pretrain_model(base_path + "/pytorch_model.bin", model, accelerator)
        
        optimizer.load_state_dict(torch.load(base_path + "/optimizer.bin"))
        scheduler.load_state_dict(torch.load(base_path + "/scheduler.bin"))
        
        accelerator.print(f'Loading training state successfully! Start training from {starting_epoch}, Best score: {best_score}')
        
        return model, optimizer, scheduler, starting_epoch, train_step, val_step, best_score, best_metrics
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'Failed to load training state!')
        return model, optimizer, scheduler, 0, 0, 0, torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False), {}


def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print(f'Successfully loaded the training model for ', pretrain_path)
        return model
    except Exception as e:
        try:
            state_dict = load_model_dict(pretrain_path)
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace('module.', '')] = state_dict[key]
            model.load_state_dict(new_state_dict)
            accelerator.print(f'Successfully loaded the training modelfor ', pretrain_path)
            return model
        except Exception as e:
            accelerator.print(e)
            accelerator.print(f'Failed to load the training model！')
            return model


def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建。")
    else:
        print(f"目录 {directory} 已存在。")

def get_directory_item(path):
    try:
        # 获取指定路径下的所有条目
        entries = os.listdir(path)
        return entries
    except FileNotFoundError:
        print(f"The provided path {path} does not exist.")
        return []

def write_result(config, path, result):
    directory = os.path.dirname(path)
    ensure_directory_exists(directory)
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        pass  
    
    cannot_open, loss_model, unalign_model, unResolution_model, unSize_model, use_model = result
    
    result_line = []
    with open(path, 'w') as f:
        # 1. 写入可用数据编号
        result = ''
        for item in use_model:
            result += item + ', '
        result_line.append('Useful data: \n' + result.rstrip(', ') + '\n')
        result_line.append('\n')
        
        # 2. 无法打开的文件（文件损坏）
        result_line.append('Unable data: '+ '\n')
        for key in cannot_open.keys():
            result = ''
            data = cannot_open[key]
            for item in data:
                result += item + ', '
            result = result.rstrip(', ')
            result_line.append(f'{key}: {result} \n')
        result_line.append('\n')
        
        # 2. 写入缺失模态的编号
        result_line.append('Loss models data: '+ '\n')
        for key in loss_model.keys():
            result = ''
            data = loss_model[key]
            for item in data:
                result += item + ', '
            result = result.rstrip(', ')
            result_line.append(f'{key}: {result} \n')
        result_line.append('\n')  
        
        # 3. 写入不对齐模态的编号
        result_line.append('Unalign data: '+ '\n')
        for key in unalign_model.keys():
            result = ''
            data = unalign_model[key]
            for key2 in data.keys():
                result += f'{key2} : {data[key2]}' + ', '
            result = result.rstrip(', ')
            result_line.append(f'{key}: {result} \n')    
        result_line.append('\n') 
        
        # 4. 写入分辨率不足模态的编号
        result_line.append(f'unResolution data ({config.lowestResolution}): '+ '\n')
        for key in unResolution_model.keys():
            result = ''
            data = unResolution_model[key]
            for key2 in data.keys():
                result += f'{key2} : {data[key2]}' + ', '
            result = result.rstrip(', ')
            result_line.append(f'{key}: {result} \n')    
        result_line.append('\n') 

        for line in result_line:
            f.write(line)
        
        # 5. 写入病灶过小的编号
        result_line.append(f'unSize data ({config.lowestSize}): '+ '\n')
        for key in unSize_model.keys():
            result = ''
            data = unSize_model[key]
            for key2 in data.keys():
                result += f'{key2} : {data[key2]}' + ', '
            result = result.rstrip(', ')
            result_line.append(f'{key}: {result} \n')    
        result_line.append('\n') 

        for line in result_line:
            f.write(line)


def check_open_files(main_directory, checkModels):
    error = []
    for model in checkModels:
        modality_dir = os.path.join(main_directory, model)
        if not os.path.isdir(modality_dir):
            continue
        
        nii_files = [f for f in os.listdir(modality_dir) if f.endswith('.nii.gz')]
        if not nii_files:
            continue
        
        try:
            nii_file_path = os.path.join(modality_dir, nii_files[0])
            img = nib.load(nii_file_path)
        except Exception as e:
            print(e)
            error.append(model)
    return (not error, error)
             

def check_subdirectories_contain_files(main_directory, subdirectory_names, checkModels):
    empty_dirs = []
    entries = os.listdir(main_directory)
    # 判断要检查的模态是否全部包含在目录中
    for model in checkModels:
        if model not in entries:
            empty_dirs.append(model)
            
    for subdir_name in subdirectory_names:
        if subdir_name not in checkModels:
            continue
        else:
            subdir_path = os.path.join(main_directory, subdir_name)
            if not os.path.isdir(subdir_path):
                # 如果不是有效的目录，则添加到空目录列表并继续下一个
                empty_dirs.append(subdir_name)
                continue
            
            # 获取子目录中的所有条目，并过滤出文件
            files_in_subdir = [f for f in os.listdir(subdir_path) 
                            if os.path.isfile(os.path.join(subdir_path, f))]
            
            if not files_in_subdir:
                empty_dirs.append(subdir_name)
    
    # 如果empty_dirs为空，说明所有子目录都包含文件
    return (not empty_dirs, empty_dirs)

def check_slices_consistency(main_directory, checkModels):
    slice_counts = {}
    inconsistent_modalities = {}

    # 遍历所有模态子目录
    for modality in checkModels:
        modality_dir = os.path.join(main_directory, modality)
        if not os.path.isdir(modality_dir):
            print(f"警告：未找到 {modality} 模态的子目录")
            continue
        
        nii_files = [f for f in os.listdir(modality_dir) if f.endswith('.nii.gz')]
        if not nii_files:
            print(f"警告：{modality} 模态子目录下没有 .nii.gz 文件")
            continue
        
        # 假设每个模态只有一个 .nii.gz 文件，如果有多个，请根据需要调整逻辑
        nii_file_path = os.path.join(modality_dir, nii_files[0])
        
        # 加载.nii.gz文件并获取2D切片数量
        img = nib.load(nii_file_path)
        slices = img.shape[2] if len(img.shape) >= 3 else None  # 假设第三维为切片维度
        
        if slices is None:
            print(f"警告：无法确定 {modality} 模态文件的2D切片数量")
            continue
        
        slice_counts[modality] = slices
    
    # 检查所有模态的2D切片数量是否一致
    reference_slices = next(iter(slice_counts.values()), None) if slice_counts else None
    for modality, slices in slice_counts.items():
        if slices != reference_slices:
            inconsistent_modalities[modality] = slices
    
    return (not bool(inconsistent_modalities), inconsistent_modalities)

def check_modalities_resolution(main_directory, checkModels, lowestResolution):
    inadequate_resolutions = {}

    for modality in checkModels:
        modality_dir = os.path.join(main_directory, modality)
        if not os.path.isdir(modality_dir):
            continue
        
        nii_files = [f for f in os.listdir(modality_dir) if f.endswith('.nii.gz')]
        if not nii_files:
            continue
        
        # 假设每个模态只有一个 .nii.gz 文件，如果有多个，请根据需要调整逻辑
        nii_file_path = os.path.join(modality_dir, nii_files[0])
        
        try:
            # 加载.nii.gz文件并获取分辨率
            img = nib.load(nii_file_path)
            resolution = img.shape[:3]  # 获取体素大小
            
            # 检查前两维分辨率是否都大于200
            if len(resolution) < 2 or resolution[0] < lowestResolution[0] or resolution[1] < lowestResolution[1]:
                inadequate_resolutions[modality] = resolution
        except Exception as e:
            inadequate_resolutions[modality] = None  # 或者其他适当的默认值

    return (not bool(inadequate_resolutions), inadequate_resolutions)

def check_label_size(main_directory, checkModels, lowestSize):
    inaccaptable_size = {}

    for modality in checkModels:
        modality_dir = os.path.join(main_directory, modality)
        if not os.path.isdir(modality_dir):
            continue
        
        nii_files = [f for f in os.listdir(modality_dir) if f.endswith('seg.nii.gz')]
        if not nii_files:
            continue
        
        # 假设每个模态只有一个 .nii.gz 文件，如果有多个，请根据需要调整逻辑
        nii_file_path = os.path.join(modality_dir, nii_files[0])
        
        try:
            # 加载.nii.gz文件并获取分辨率
            img = nib.load(nii_file_path)
            size = np.count_nonzero(img.get_fdata())  # 获取体素大小
            
            # 检查前两维分辨率是否都大于200
            if size < lowestSize:
                inaccaptable_size[modality] = size
        except Exception as e:
            inaccaptable_size[modality] = None  # 或者其他适当的默认值

    return (not bool(inaccaptable_size), inaccaptable_size)


def check(config, dataPath, checkModels):
    directory_item = get_directory_item(dataPath)

    # TODO: 0. 判断是否有打不开的nii文件
    cannot_open = {}
    # TODO: 1. 确定哪个编号空缺模态，如皆不空缺，返回(True, []), 否则返回(False, [模态名])
    loss_model = {}
    # TODO: 2. 确定同编号下，不同模态切片数量是否对齐，如皆对齐返回(True, {}), 如不对齐返回(False, {模态名：切片数量})
    unalign_model = {}
    # TODO: 3. 确定所有模态分辨率是否合理，合理返回(True, {}), 不合理返回(False, {模态名：分辨率})
    unResolution_model = {}
    # TODO: 4. 确定所有模态的病灶大小是否合理，合理返回(True, {}), 不合理返回(False, {模态名：病灶体素大小})
    unSize_model = {}
    # TODO: 5. 统筹能用的数据编号
    use_model = []
    if directory_item != []:
        for item in directory_item:
            this_data_path = dataPath + '/' + item + '/'
            all_models = get_directory_item(this_data_path)
            # TODO: 0
            check_flag, error_item = check_open_files(this_data_path, checkModels)
            if check_flag == False:
                cannot_open[item] = error_item
                continue
            # TODO: 1
            check_flag, loss_item = check_subdirectories_contain_files(this_data_path, all_models, checkModels)
            if check_flag == False:
                loss_model[item] = loss_item
            else:
                # 如果模态缺失，则无所谓对齐，如果不缺失，则考虑对齐
                # TODO: 2
                check_flag, unalign_item = check_slices_consistency(this_data_path, checkModels)
                if check_flag == False:
                    unalign_model[item] = unalign_item
        
            # 无论模态是否缺失，判断已有模态分辨率是否达标
            # TODO: 3
            check_flag, unResolution_item = check_modalities_resolution(this_data_path, checkModels, config.lowestResolution)
            
            if check_flag  == False:
                unResolution_model[item] = unResolution_item
            
            # 无论模态是否缺失，判断已有模态病灶大小是否达标
            # TODO: 4
            check_flag, unSize_item = check_label_size(this_data_path, checkModels, config.lowestSize)

            if check_flag  == False:
                unSize_model[item] = unSize_item
            
    else:
        print('dataPath has not data!')
      
    # TODO: 5
    for item in directory_item:
        if item not in loss_model.keys() and item not in unalign_model.keys() and item not in unResolution_model.keys() and item not in unSize_model.keys():
            use_model.append(item)
  
    return cannot_open, loss_model, unalign_model, unResolution_model, unSize_model, use_model

def data_check(config):
    # 检查的模态
    checkModels = config.data_check.checkModels
    # 写入路径
    writePath = config.data_check.writePath
  
    # 非手术勾画
    dataPath1 = config.data_check.dataPath1
    all_result1 = check(config.data_check, dataPath1, checkModels)
    write_result(config.data_check, writePath+'/'+'NonsurgicalMR.txt', all_result1)
  
    # 手术勾画
    dataPath2 = config.data_check.dataPath2
    all_result2 = check(config.data_check, dataPath2, checkModels)
    write_result(config.data_check, writePath+'/'+'SurgicalMR.txt', all_result2)

@torch.no_grad()
def visualize_for_all(config, image_list, model, accelerator, load_transform):
    model.eval()
    for i in range(len(image_list)):
        visualization_choose_image = image_list[i]['image'][0].split('/')[-1].replace('.nii.gz','')
        visualization_choose_dir   = ''
        for word in image_list[i]['image'][0].split('/'):
            if 'urgical' in word:
                visualization_choose_dir   = word
        choose_image = config.loader.dataPath + '/' + visualization_choose_dir + '/' + f'{visualization_choose_image}'
        
        
        accelerator.print('visualize for image: ', choose_image)

    
        images = []
        labels = []
        image_size = []
        affines = []
        for i in range(len(config.loader.checkModels)):
            image_path = choose_image + '/' + config.loader.checkModels[i] + '/' + f'{visualization_choose_image}.nii.gz'
            label_path = choose_image + '/' + config.loader.checkModels[i] + '/' + f'{visualization_choose_image}seg.nii.gz'
            
            batch = load_transform[i]({
                'image': image_path,
                'label': label_path
            })
            images.append(batch['image'].unsqueeze(1))
            labels.append(batch['label'].unsqueeze(1))
            image_size.append(tuple(batch['image_meta_dict']['spatial_shape'][i].item() for i in range(3)))
            affines.append(batch['label_meta_dict']['affine'])
            
        image_tensor = torch.cat(images, dim=1)
        label_tensor = torch.cat(labels, dim=1)
        
        inference = monai.inferers.SlidingWindowInferer(roi_size=config.loader.target_size, overlap=0.5,
                                                        sw_device=accelerator.device, device=accelerator.device)
        post_trans = monai.transforms.Compose([
            monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
        ])
        
        img = inference(image_tensor.to(accelerator.device), model.to(accelerator.device))
        seg = post_trans(img[0])
        
        for i in range(len(config.loader.checkModels)):
            seg_now = monai.transforms.Resize(spatial_size=image_size[i], mode=("nearest-exact"))(seg[i].unsqueeze(0))
            seg_now = seg_now[0]
            affine = affines[i]
            seg_out = np.zeros((seg_now.shape[0], seg_now.shape[1], seg_now.shape[2]))
            
            seg_now = seg_now.cpu()
            seg_out[seg_now==1] = 1
            res = nib.Nifti1Image(seg_out.astype(np.uint8), affine)
            
            save_path  = config.visualization.image_path + '/' + config.visualization.choose_dir + '/' + f'{config.visualization.choose_image}' + '/' + config.loader.checkModels[i]
            ensure_directory_exists(save_path)
            picture = nib.load(choose_image + '/' + config.loader.checkModels[i] + '/' + f'{visualization_choose_image}seg.nii.gz')
            
            qform = picture.get_qform()
            res.set_qform(qform)
            sfrom = picture.get_sform()
            res.set_sform(sfrom)
            
            original_str = f"{save_path}/{visualization_choose_image}inference.nii.gz"

            print('save ', original_str)
            # 然后保存 NIFTI 图像
            nib.save(
                res,
                original_str,
            )