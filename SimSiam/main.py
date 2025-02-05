import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
import wandb
from custom_loader import create_dataloader
import random

def plot_images(images1, images2, labels, name):
    import matplotlib.pyplot as plt
    
    # Take first 8 pairs of images
    n = 8
    images1 = images1[:n]
    images2 = images2[:n]
    labels = labels[:n]
    
    # Create a figure with n rows and 3 columns
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    
    # Define normalization constants
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    
    # For each row, plot image1, image2 and label
    for i in range(n):
        # Unnormalize images (multiply by std and add mean)
        img1 = images1[i] * std + mean
        img2 = images2[i] * std + mean
        
        # Convert tensors to numpy arrays and transpose to correct format (H,W,C)
        img1 = img1.cpu().numpy().transpose(1,2,0)
        img2 = img2.cpu().numpy().transpose(1,2,0)
        
        # Clip values to [0,1] range
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        # Plot images and label
        axes[i,0].imshow(img1)
        axes[i,0].axis('off')
        axes[i,0].set_title('Image 1')
        
        axes[i,1].imshow(img2)
        axes[i,1].axis('off')
        axes[i,1].set_title('Image 2')
        
        axes[i,2].text(0.5, 0.5, f'Label: {labels[i].item()}', 
                      horizontalalignment='center',
                      verticalalignment='center')
        axes[i,2].axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the figure with timestamp
    plt.savefig(f'{name}_image_pairs.png')
    plt.close()

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(device, args):
    # Set random seed first thing in main
    set_seed(4)
    
    DEBUG = False
    LABEL_RANGE = [0, 5] #Range of images is 0 to 5.
    AUGMENT_IMAGES = False
    # name = 'no_augment' if not AUGMENT_IMAGES else 'augment'
    # name += str(LABEL_RANGE[0]) + '_' + str(LABEL_RANGE[1])
    name = 'non_symmetric_loss'

    wandb.init(project="SimSiamRight", name=name, config=args, mode='disabled' if DEBUG else 'online')

    args.model.name = 'simsiam_diffusion'

    train_loader = create_dataloader(
        h5_path="new_cifar10_dataset.h5",
        batch_size=args.train.batch_size,
        shuffle=True,
        transform=get_aug(train=True, image_size=args.aug_kwargs['image_size'], name='simsiam_diffusion'),
        augment_images=AUGMENT_IMAGES,
        num_workers=8,
        range_of_images=LABEL_RANGE #Set as -1 to use all synthetic images. 
    )

    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader) // 5,
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    accuracy = 0 
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch // 5), desc=f'Training')
    for epoch in global_progress:
        model.train()
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            labels = labels / (LABEL_RANGE[1] + 1) #Normalize labels to be between 0 and 1. 
            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True), labels.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            wandb.log(data_dict)
            
            local_progress.set_postfix(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 
        
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        wandb.log(epoch_dict)
        global_progress.set_postfix(epoch_dict)
    
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth")
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














