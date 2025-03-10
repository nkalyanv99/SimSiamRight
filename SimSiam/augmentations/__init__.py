from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .simsiam_diffusion_aug import SimSiamDiffusionTransform

def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None):

    if train==True:
        if name =='simsiam_diffusion':
            augmentation = SimSiamDiffusionTransform(image_size)
        elif name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








