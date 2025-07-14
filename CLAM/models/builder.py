import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = True
    UNI_CKPT_PATH = '/mnt/ncshare/ozkilim/BRCA/data/UMI/ckpts/vit_large_patch16_224.dinov2.uni_mass100k'
    # check if UNI_CKPT_PATH is set, catch exception if not
    # try:
    #     # check if UNI_CKPT_PATH is set
    #     print(os.environ)
    #     if 'UNI_CKPT_PATH' not in os.environ:
    #         raise ValueError('UNI_CKPT_PATH not set')
    #     HAS_UNI = True
    #     UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    # except Exception as e:
    #     print(e)
    return HAS_UNI, UNI_CKPT_PATH
        

def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    
    img_transforms = None  # Initialize img_transforms
    
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                             std=constants['std'],
                                             target_img_size=target_img_size)
    
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load("resources/pytorch_model_UNI.bin", map_location="cpu"), strict=True)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                             std=constants['std'],
                                             target_img_size=target_img_size)

    elif model_name == 'uni_v2':
        
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
                }
        
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        img_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    

    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                             std=constants['std'],
                                             target_img_size=target_img_size)
    

    elif model_name == 'conch_v1_5':
        from transformers import AutoModel
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, img_transforms = titan.return_conch()
        model.eval()


    elif model_name == 'prov_giga_path':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        img_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    elif model_name == 'virchow':
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = model.eval()
        img_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    elif model_name == 'virchow_v2':
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = model.eval()
        img_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    elif model_name == 'H-optimus-0':
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        model = model.eval()
        img_transforms = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
    
    else:
        raise NotImplementedError('Model {} not implemented'.format(model_name))
    
    return model, img_transforms
