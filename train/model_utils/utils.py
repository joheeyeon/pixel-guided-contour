import torch
import os
import torch.nn.functional
from termcolor import colored

def load_model(net, optim, scheduler, recorder, model_path, map_location=None):
    strict = True
    if os.path.isdir(model_path):
        model_path = f"{model_path}/latest.pth"

    if not os.path.exists(model_path):
        print(colored(f'WARNING: NO MODEL LOADED !!! {model_path}', 'red'))
        return 0, None, 0, 0, None

    print('load model: {}'.format(model_path))
    if map_location is None:
        pretrained_model = torch.load(model_path, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                                'cuda:2': 'cpu', 'cuda:3': 'cpu'})
    else:
        pretrained_model = torch.load(model_path, map_location=map_location)

    if hasattr(net, 'module'):
        net.module.load_state_dict(pretrained_model['net'], strict=strict)
    else:
        net.load_state_dict(pretrained_model['net'], strict=strict)
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    if 'best_ap' in pretrained_model:
        best_val_ap = pretrained_model['best_ap']
    else:
        best_val_ap = None
    if 'patience' in pretrained_model:
        patience = pretrained_model['patience']
    else:
        patience = 0
    if 'best_epoch' in pretrained_model:
        best_epoch = pretrained_model['best_epoch']
    else:
        best_epoch = pretrained_model['epoch'] - patience
    if 'add_iou_loss' in pretrained_model:
        add_iou_loss = pretrained_model['add_iou_loss']
    else:
        add_iou_loss = None

    return pretrained_model['epoch'] + 1, best_val_ap, patience, best_epoch, add_iou_loss

def save_model(net, optim, scheduler, recorder, epoch, model_dir, tag=None, best_val_ap=None, patience=None, best_epoch=None, add_iou_loss=None):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch,
        'best_ap': best_val_ap,
        'patience': patience,
        'best_epoch': best_epoch,
        'add_iou_loss': add_iou_loss
    }, os.path.join(model_dir, '{}.pth'.format(epoch if tag is None else tag)))
    return

def save_weight(net, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
    }, os.path.join(model_dir, '{}.pth'.format('final')))
    return

def load_network(net, model_dir, strict=True, map_location=None, specific=None):

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_dir))
    if map_location is None:
        pretrained_model = torch.load(model_dir, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                               'cuda:2': 'cpu', 'cuda:3': 'cpu'})
    else:
        pretrained_model = torch.load(model_dir, map_location=map_location)
    if 'epoch' in pretrained_model.keys():
        epoch = pretrained_model['epoch']
    else:
        epoch = 0

    if 'net' in pretrained_model:
        pretrained_model = pretrained_model['net']
    elif 'state_dict' in pretrained_model:
        pretrained_model = pretrained_model['state_dict']

    net_weight = net.state_dict()
    for key in net_weight.keys():
        if key in pretrained_model:
            if list(net_weight[key].size())==list(pretrained_model[key].size()):
                net_weight.update({key: pretrained_model[key]})

    if hasattr(net, 'module'):
        net.module.load_state_dict(net_weight, strict=strict)
    else:
        net.load_state_dict(net_weight, strict=strict)
    return epoch

def load_network_lightning(net, model_dir, strict=True):
    """
    Function to load checkpoints saved with PyTorch Lightning.
    Extracts weights from 'state_dict' and automatically removes prefixes like 'network.'.
    """
    if not os.path.exists(model_dir):
        print(colored(f'WARNING: Lightning checkpoint not found at {model_dir}', 'red'))
        return 0

    print(f'Loading Lightning model: {model_dir}')

    # Load checkpoint to CPU first for stability
    checkpoint = torch.load(model_dir, map_location='cpu')

    # 1. Extract actual weights (state_dict) from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        print(colored("WARNING: 'state_dict' key not found. Assuming the file is a raw state_dict.", 'yellow'))
        state_dict = checkpoint

    # 2. Remove various prefixes like 'loss_module.net.', 'network.' to clean up keys
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('loss_module.net.'):
            new_key = k[len('loss_module.net.'):]
        elif k.startswith('network.'):
            new_key = k[len('network.'):]
        else:
            new_key = k
        cleaned_state_dict[new_key] = v

    # 3. Load cleaned weights directly into the model
    net.load_state_dict(cleaned_state_dict, strict=strict)

    return checkpoint.get('epoch', 0)

def load_stage1_modules_selective(net, model_path, cfg=None):
    """
    Function to selectively load modules trained in Stage 1.
    Even if the entire model structure differs, this function compares only Stage 1 modules
    and loads only the compatible parts.
    
    Args:
        net: Current network (Stage 2 structure)
        model_path: Path to Stage 1 model
        cfg: Configuration object (for checking stage1_train_ct_hm, stage1_train_wh)
    
    Returns:
        bool: Whether loading was successful
    """
    if not os.path.exists(model_path):
        print(colored(f'WARNING: Stage 1 model not found at {model_path}', 'red'))
        return False
    
    try:
        print(f'[SELECTIVE LOAD] Loading Stage 1 modules from: {model_path}')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state_dict
        if 'state_dict' in checkpoint:
            stage1_state_dict = checkpoint['state_dict']
        elif 'net' in checkpoint:
            stage1_state_dict = checkpoint['net']
        else:
            stage1_state_dict = checkpoint
        
        # Clean up prefixes
        cleaned_stage1_dict = {}
        for k, v in stage1_state_dict.items():
            if k.startswith('loss_module.net.'):
                new_key = k[len('loss_module.net.'):]
            elif k.startswith('network.'):
                new_key = k[len('network.'):]
            else:
                new_key = k
            cleaned_stage1_dict[new_key] = v
        
        # Get state_dict of current model
        current_state_dict = net.state_dict()
        
        # Define modules trained in Stage 1
        stage1_trained_ct_hm = getattr(cfg.train, 'stage1_train_ct_hm', False) if cfg else False
        stage1_trained_wh = getattr(cfg.train, 'stage1_train_wh', False) if cfg else False
        
        # Stage 1 module patterns
        stage1_module_patterns = [
            'pixel',      # pixel head (always included)
            'dla',        # backbone
            'backbone',   # backbone
            'base_layer', # backbone
        ]
        
        # Add additional modules based on options
        if stage1_trained_ct_hm:
            stage1_module_patterns.append('ct_hm')
        if stage1_trained_wh:
            stage1_module_patterns.append('wh')
        
        # Perform selective loading
        loaded_params = 0
        total_stage1_params = 0
        skipped_params = []
        
        for param_name, param_value in cleaned_stage1_dict.items():
            # Check if it's a Stage 1 module
            is_stage1_module = any(pattern in param_name.lower() for pattern in stage1_module_patterns)
            
            if is_stage1_module:
                total_stage1_params += 1
                
                # Check if the parameter exists in the current model and has matching size
                if param_name in current_state_dict:
                    if current_state_dict[param_name].shape == param_value.shape:
                        # Successfully loaded
                        current_state_dict[param_name] = param_value
                        loaded_params += 1
                        print(f"  ✅ [LOADED] {param_name} {param_value.shape}")
                    else:
                        # Shape mismatch
                        skipped_params.append(f"{param_name} (shape mismatch: current={current_state_dict[param_name].shape}, stage1={param_value.shape})")
                        print(f"  ❌ [SKIP] {param_name} - shape mismatch: current={current_state_dict[param_name].shape}, stage1={param_value.shape}")
                else:
                    # Parameter not found
                    skipped_params.append(f"{param_name} (not found in current model)")
                    print(f"  ❌ [SKIP] {param_name} - not found in current model")
        
        # Load updated state_dict
        net.load_state_dict(current_state_dict, strict=True)
        
        print(f"[SELECTIVE LOAD] Summary:")
        print(f"  Stage 1 modules found: {total_stage1_params}")
        print(f"  Successfully loaded: {loaded_params}")
        print(f"  Skipped: {len(skipped_params)}")
        
        if skipped_params:
            print(f"  Skipped parameters:")
            for skip_info in skipped_params[:5]:  # Print at most 5
                print(f"    - {skip_info}")
            if len(skipped_params) > 5:
                print(f"    - ... and {len(skipped_params) - 5} more")
        
        if loaded_params > 0:
            print(f"[SELECTIVE LOAD] ✅ Successfully loaded {loaded_params} Stage 1 parameters")
            return True
        else:
            print(f"[SELECTIVE LOAD] ❌ No parameters could be loaded")
            return False
            
    except Exception as e:
        print(colored(f'ERROR: Failed to load Stage 1 modules: {str(e)}', 'red'))
        return False
