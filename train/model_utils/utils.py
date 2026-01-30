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
    PyTorch Lightning으로 저장된 체크포인트(.ckpt)를 불러오기 위한 전용 함수.
    'state_dict'에서 가중치를 추출하고, 'network.' 같은 접두사를 자동으로 제거합니다.
    """
    if not os.path.exists(model_dir):
        print(colored(f'WARNING: Lightning checkpoint not found at {model_dir}', 'red'))
        return 0

    print(f'Loading Lightning model: {model_dir}')

    # 체크포인트 파일을 CPU로 먼저 불러와 안정성을 높입니다.
    checkpoint = torch.load(model_dir, map_location='cpu')

    # 1. 체크포인트에서 실제 가중치(state_dict)를 추출합니다.
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        print(colored("WARNING: 'state_dict' key not found. Assuming the file is a raw state_dict.", 'yellow'))
        state_dict = checkpoint

    # 2. 'loss_module.net.', 'network.' 등 다양한 접두사를 제거하여 키를 정리합니다.
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('loss_module.net.'):
            new_key = k[len('loss_module.net.'):]
        elif k.startswith('network.'):
            new_key = k[len('network.'):]
        else:
            new_key = k
        cleaned_state_dict[new_key] = v

    # 3. 정리된 가중치를 모델에 직접 로드합니다.
    net.load_state_dict(cleaned_state_dict, strict=strict)

    return checkpoint.get('epoch', 0)

def load_stage1_modules_selective(net, model_path, cfg=None):
    """
    Stage 1에서 학습된 모듈만 선택적으로 로드하는 함수.
    전체 모델 구조가 다르더라도 Stage 1 모듈들만 비교해서 호환되는 부분만 로드합니다.
    
    Args:
        net: 현재 네트워크 (Stage 2 구조)
        model_path: Stage 1 모델 경로
        cfg: 설정 객체 (stage1_train_ct_hm, stage1_train_wh 확인용)
    
    Returns:
        bool: 로드 성공 여부
    """
    if not os.path.exists(model_path):
        print(colored(f'WARNING: Stage 1 model not found at {model_path}', 'red'))
        return False
    
    try:
        print(f'[SELECTIVE LOAD] Loading Stage 1 modules from: {model_path}')
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # state_dict 추출
        if 'state_dict' in checkpoint:
            stage1_state_dict = checkpoint['state_dict']
        elif 'net' in checkpoint:
            stage1_state_dict = checkpoint['net']
        else:
            stage1_state_dict = checkpoint
        
        # 접두사 정리
        cleaned_stage1_dict = {}
        for k, v in stage1_state_dict.items():
            if k.startswith('loss_module.net.'):
                new_key = k[len('loss_module.net.'):]
            elif k.startswith('network.'):
                new_key = k[len('network.'):]
            else:
                new_key = k
            cleaned_stage1_dict[new_key] = v
        
        # 현재 모델의 state_dict
        current_state_dict = net.state_dict()
        
        # Stage 1에서 학습된 모듈들 정의
        stage1_trained_ct_hm = getattr(cfg.train, 'stage1_train_ct_hm', False) if cfg else False
        stage1_trained_wh = getattr(cfg.train, 'stage1_train_wh', False) if cfg else False
        
        # Stage 1 모듈 패턴들
        stage1_module_patterns = [
            'pixel',      # pixel head (항상 포함)
            'dla',        # backbone
            'backbone',   # backbone
            'base_layer', # backbone
        ]
        
        # 옵션에 따라 추가 모듈
        if stage1_trained_ct_hm:
            stage1_module_patterns.append('ct_hm')
        if stage1_trained_wh:
            stage1_module_patterns.append('wh')
        
        # 선택적 로드 수행
        loaded_params = 0
        total_stage1_params = 0
        skipped_params = []
        
        for param_name, param_value in cleaned_stage1_dict.items():
            # Stage 1 모듈인지 확인
            is_stage1_module = any(pattern in param_name.lower() for pattern in stage1_module_patterns)
            
            if is_stage1_module:
                total_stage1_params += 1
                
                # 현재 모델에 해당 파라미터가 있고 크기가 일치하는지 확인
                if param_name in current_state_dict:
                    if current_state_dict[param_name].shape == param_value.shape:
                        # 로드 성공
                        current_state_dict[param_name] = param_value
                        loaded_params += 1
                        print(f"  ✅ [LOADED] {param_name} {param_value.shape}")
                    else:
                        # 크기 불일치
                        skipped_params.append(f"{param_name} (shape mismatch: current={current_state_dict[param_name].shape}, stage1={param_value.shape})")
                        print(f"  ❌ [SKIP] {param_name} - shape mismatch: current={current_state_dict[param_name].shape}, stage1={param_value.shape}")
                else:
                    # 파라미터 없음
                    skipped_params.append(f"{param_name} (not found in current model)")
                    print(f"  ❌ [SKIP] {param_name} - not found in current model")
        
        # 업데이트된 state_dict 로드
        net.load_state_dict(current_state_dict, strict=True)
        
        print(f"[SELECTIVE LOAD] Summary:")
        print(f"  Stage 1 modules found: {total_stage1_params}")
        print(f"  Successfully loaded: {loaded_params}")
        print(f"  Skipped: {len(skipped_params)}")
        
        if skipped_params:
            print(f"  Skipped parameters:")
            for skip_info in skipped_params[:5]:  # 최대 5개만 출력
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
