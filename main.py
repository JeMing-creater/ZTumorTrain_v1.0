import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader,read_usedata,load_MR_dataset_images
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, resume_train_state, data_check

from src.model.HWAUNETR import HWAUNETR
from visualization import visualize_for_all
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
          metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
          post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch['image'])
        total_loss = 0
        log = ''
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch['label'])
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += alpth * loss
        # val_outputs = [post_trans(i) for i in logits]
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.backward(total_loss)
        # if accelerator.sync_gradients:
        #     accelerator.clip_grad_norm_(
        #         model.parameters(), 1.0)
        #     )
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
            flush=True
            )
        step += 1
    scheduler.step(epoch)
    metric = {} 
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
            f'Train/mean {metric_name}': float(batch_acc.mean()),
            f'Train/Object1 {metric_name}': float(batch_acc[0]),
            f'Train/Object2 {metric_name}': float(batch_acc[1])
        })
    accelerator.log(metric, step=epoch)
    return step

@torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator):
    # 验证
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch['image'], model)
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.print(
            f'[{i + 1}/{len(val_loader)}] Validation Loading...',
            flush=True)
        
        step += 1
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0]
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc.to(accelerator.device)) / accelerator.num_processes
        metrics[metric_name].reset()
        if metric_name == 'dice_metric':
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/Object1 {metric_name}': float(batch_acc[0]),
                f'Val/Object2 {metric_name}': float(batch_acc[1])
            })
            dice_acc = torch.Tensor([metric['Val/mean dice_metric']]).to(accelerator.device)
            dice_class = batch_acc
        else:
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/Object1 {metric_name}': float(batch_acc[0]),
                f'Val/Object2 {metric_name}': float(batch_acc[1])
            })
            hd95_acc = torch.Tensor([metric['Val/mean hd95_metric']]).to(accelerator.device)
            hd95_class = batch_acc
    return dice_acc, dice_class, hd95_acc, hd95_class, step


if __name__ == '__main__':
    print('Program star!')
    
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    
    
    dataPath = input("Please input data path: ")
    image_path = input("Please input image save path: ")
    
    config.loader.dataPath = dataPath
    config.data_check.dataPath1 = dataPath + '/' + 'NonsurgicalMR'
    config.data_check.dataPath2 = dataPath + '/' + 'SurgicalMR'
    config.data_check.writePath = dataPath
    config.visualization.image_path = image_path
    
    print('Data checking!')
    data_check(config)
    
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))
    
    accelerator.print('load model...')
    model = HWAUNETR(in_chans=2, out_chans=2, fussion = [1, 2, 4, 8], kernel_sizes=[4, 2, 2, 2], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8], out_indices=[0, 1, 2, 3])
    
    accelerator.print('load dataset...')
    train_loader, val_loader = get_dataloader(config)
    
    inference = monai.inferers.SlidingWindowInferer(roi_size=config.loader.target_size, overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=True,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    
    
    # start training
    accelerator.print("Start Training！")
    train_step = 0
    best_eopch = -1
    val_step = 0
    best_score = 0
    start_num_epochs = 0
    best_metrics = []
    
    if config.trainer.resume:
        model, optimizer, scheduler, start_num_epochs, train_step, val_step, best_score, best_metrics = resume_train_state(model, config.finetune.checkpoint, optimizer, scheduler, accelerator)
        
    
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)
    
    best_score = torch.Tensor([best_score]).to(accelerator.device)
    
    for epoch in range(start_num_epochs, config.trainer.num_epochs):
        train_step = train_one_epoch(model, loss_functions, train_loader,
                     optimizer, scheduler, metrics,
                     post_trans, accelerator, epoch, train_step)

        dice_acc, dice_class, hd95_acc, hd95_class, val_step = val_one_epoch(model, inference, val_loader,
                                                                   metrics, val_step,
                                                                   post_trans, accelerator)
        print(f'Epoch [{epoch}/{config.trainer.num_epochs}] dice acc: {dice_acc} best acc: {best_score}')
        
        if dice_acc > best_score:
            best_score = dice_acc
            best_metrics = dice_class
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/")
                torch.save(model.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/model.pth")
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/")
            torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/epoch.pth.tar')
    
    accelerator.print(f"dice score: {best_score}")
    accelerator.print(f"dice metrics : {best_metrics}")
    
    datapath = config.loader.dataPath
    use_models = config.loader.checkModels
    
    datapath1 = datapath + '/' + 'NonsurgicalMR' + '/'
    datapath2 = datapath + '/' + 'SurgicalMR' + '/'
    usedata1 = datapath + '/' + 'NonsurgicalMR.txt'
    usedata2 = datapath + '/' + 'SurgicalMR.txt'
    usedata1 = read_usedata(usedata1)
    usedata2 = read_usedata(usedata2)
    
    data1 = load_MR_dataset_images(datapath1, usedata1, use_models)
    data2 = load_MR_dataset_images(datapath2, usedata2, use_models)
    
    data = data1 + data2
    
    visualize_for_all(config=config, image_list=data, model = model, accelerator = accelerator)
    sys.exit(1)

