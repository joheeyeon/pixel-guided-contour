import os
from .trainer import Trainer
import torch


def _wrapper_factory(network, cfg):
    if cfg.commen.task.split('+')[0] == 'ccp':
        from .ccp import NetworkWrapper
    else:
        raise ValueError(f"Unsupported task: {cfg.commen.task}")
    
    wrapper = NetworkWrapper(network, with_dml=cfg.train.with_dml,
                          ml_start_epoch=cfg.train.ml_start_epoch, weight_dict=cfg.train.weight_dict, cfg=cfg)
    if cfg.commen.task.split('+')[0] in ['ccp', 'ccp_maskinit']:
        if hasattr(cfg.train, 'stage') and cfg.train.stage in [1, 2]:
            apply_stage(cfg, network, wrapper)
    return wrapper


def make_trainer(network, cfg, network_t=None):
    network = _wrapper_factory(network, cfg)
    if (torch.cuda.device_count() > 1) and (cfg.train.use_dp):
        devices = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i) == torch.cuda.get_device_name(torch.cuda.current_device())]
        network = torch.nn.DataParallel(network, device_ids=devices, output_device=devices[-1])
    return Trainer(network, cfg, network_t=network_t)

def apply_stage(cfg, net, net_wrapper):
    """
    Stage별 학습/로스 세팅.
    - net_wrapper: NetworkWrapper 인스턴스(로스 weight_dict 갖고 있음)
    """
    stage = int(cfg.train.stage)

    if stage == 1:
        # 1) 파라미터: 픽셀 헤드만 학습 (+ 옵션에 따라 ct_hm, wh head도 함께)
        enable_pixel_head_only(net, cfg)

        # 2) evolve 루프 끄기(속도↑)
        cfg.model.evolve_iters = 0

        # 3) refine 등 무거운 경로 끄고 싶다면(선택)
        if hasattr(cfg.model, 'use_refine_pixel'):
            cfg.model.use_refine_pixel = False

        # 4) loss weight: pixel만 남기고 0으로 (+ ct_hm, wh head 학습 시 관련 loss도 유지)
        stage1_w = dict(net_wrapper.weight_dict)  # 복사
        train_ct_hm = getattr(cfg.train, 'stage1_train_ct_hm', False)
        train_wh = getattr(cfg.train, 'stage1_train_wh', False)
        
        for k in stage1_w.keys():
            stage1_w[k] = 0.0
        # pixel 계열만 유지 (multi-scale이면 'pixel_0','pixel_1' 등도 처리)
        for k in list(net_wrapper.weight_dict.keys()):
            if k.startswith('pixel'):
                stage1_w[k] = net_wrapper.weight_dict[k] if isinstance(net_wrapper.weight_dict[k], (int,float)) else 1.0
            # ct_hm head 학습 시 box_ct loss도 유지
            elif train_ct_hm and k == 'box_ct':
                stage1_w[k] = net_wrapper.weight_dict[k] if isinstance(net_wrapper.weight_dict[k], (int,float)) else 1.0
                print(f"[STAGE 1] CT_HM training enabled - keeping box_ct loss: {stage1_w[k]}")
            # wh head 학습 시 init loss도 유지 (wh head는 init polygon 생성에 사용)
            elif train_wh and k == 'init':
                # config에서 0으로 설정되어 있어도 Stage 1에서 wh head 학습 시 활성화
                original_weight = net_wrapper.weight_dict[k]
                stage1_w[k] = original_weight if isinstance(original_weight, (int,float)) and original_weight > 0 else 1.0
                print(f"[STAGE 1] WH training enabled - init loss: {original_weight} -> {stage1_w[k]}")
        
        # 최소 필요 항목이 빠지면 0으로 두어 안전
        stage1_w.setdefault('pixel', 1.0)
        net_wrapper.weight_dict.update(stage1_w)
        
        print(f"[STAGE 1] Final loss weights: {dict(net_wrapper.weight_dict)}")

        # (선택) stage1 용 lr - optimizer dict 내부에서 설정
        if hasattr(cfg.train, 'lr_stage1'):
            cfg.train.optimizer['lr'] = cfg.train.lr_stage1
        elif hasattr(cfg.train, 'optimizer') and 'lr_stage1' in cfg.train.optimizer:
            cfg.train.optimizer['lr'] = cfg.train.optimizer['lr_stage1']

    elif stage == 2:
        # ccp 및 ccp_maskinit task의 경우 Stage 1 모델 로드 및 freeze_s1_modules 옵션에 따라 처리  
        if 'ccp' in cfg.commen.task:
            # Stage 1 모델 로드 시도
            stage1_loaded = False
            
            # 1) config에 train.stage1_model로 path 설정된 경우
            if hasattr(cfg.train, 'stage1_model') and cfg.train.stage1_model:
                stage1_path = cfg.train.stage1_model
                if os.path.exists(stage1_path):
                    # selective loading 설정 확인
                    use_selective = getattr(cfg.train, 'use_selective_s1_loading', True)
                    if use_selective:
                        try:
                            from train.model_utils.utils import load_stage1_modules_selective
                            if load_stage1_modules_selective(net, stage1_path, cfg):
                                print(f"[STAGE 2] Selectively loaded Stage 1 modules: {stage1_path}")
                                stage1_loaded = True
                            else:
                                print(f"[STAGE 2] Selective loading failed")
                        except Exception as e:
                            print(f"[STAGE 2] Selective loading failed: {e}")
                    else:
                        print(f"[STAGE 2] Selective loading disabled by config")
            
            # 2) model_dir에서 best_s1.pth 확인
            if not stage1_loaded:
                stage1_path = os.path.join(cfg.commen.model_dir, 'best_s1.pth')
                if os.path.exists(stage1_path):
                    # selective loading 설정 확인
                    use_selective = getattr(cfg.train, 'use_selective_s1_loading', True)
                    if use_selective:
                        try:
                            from train.model_utils.utils import load_stage1_modules_selective
                            if load_stage1_modules_selective(net, stage1_path, cfg):
                                print(f"[STAGE 2] Selectively loaded Stage 1 modules from model_dir: {stage1_path}")
                                stage1_loaded = True
                            else:
                                print(f"[STAGE 2] Selective loading from model_dir failed")
                        except Exception as e:
                            print(f"[STAGE 2] Selective loading from model_dir failed: {e}")
                    else:
                        print(f"[STAGE 2] Selective loading disabled by config")
            
            # 3) Stage 1 모델 로드 실패한 경우
            if not stage1_loaded:
                print("[STAGE 2] No Stage 1 model found - training from scratch")
            
            freeze_s1 = getattr(cfg.train, 'freeze_s1_modules', False) and stage1_loaded
            task_name = cfg.commen.task.split('+')[0]
            
            if not stage1_loaded:
                # Stage 1 모델이 없으면 freeze_s1_modules 무시하고 모든 모듈 학습
                print(f"[STAGE 2] {task_name}: Training all modules (no Stage 1 model)")
                unfreeze_all(net)
            elif freeze_s1:

                stage1_trained_ct_hm = getattr(cfg.train, 'stage1_train_ct_hm', False)
                stage1_trained_wh = getattr(cfg.train, 'stage1_train_wh', False)
                
                freeze_msg = f"[STAGE 2] {task_name}: Freezing stage 1 modules (pixel head + backbone"
                if stage1_trained_ct_hm:
                    freeze_msg += " + ct_hm head"
                if stage1_trained_wh:
                    freeze_msg += " + wh head"
                freeze_msg += ")"
                print(freeze_msg)
                
                # 1) 전체 모델 freeze
                freeze_all(net)
                

                
                unfrozen_modules = []
                

                unfrozen_param_count = 0
                for name, param in net.named_parameters():
                    # ct_hm head: Stage 1에서 학습되지 않았을 때만 unfreeze
                    if 'ct_hm' in name and not stage1_trained_ct_hm:
                        param.requires_grad = True
                        unfrozen_param_count += 1
                        module_name = name.split('.')[0]
                        if module_name not in unfrozen_modules:
                            unfrozen_modules.append(module_name)
                        print(f"  [UNFREEZE] {name} (ct_hm_head)")
                    # wh head: Stage 1에서 학습되지 않았을 때만 unfreeze
                    elif 'wh' in name and not stage1_trained_wh:
                        param.requires_grad = True
                        unfrozen_param_count += 1
                        module_name = name.split('.')[0]
                        if module_name not in unfrozen_modules:
                            unfrozen_modules.append(module_name)
                        print(f"  [UNFREEZE] {name} (wh_head)")
                    # pixel refine 모듈 unfreeze (pixel head는 freeze 유지)
                    elif 'refine' in name.lower() and 'pixel' in name.lower():
                        param.requires_grad = True
                        unfrozen_param_count += 1
                        module_name = '.'.join(name.split('.')[:2]) if '.' in name else name.split('.')[0]
                        if module_name not in unfrozen_modules:
                            unfrozen_modules.append(module_name)
                        print(f"  [UNFREEZE] {name} (pixel_refine)")
                    # train_decoder 모듈 unfreeze (Stage 2에서 학습)
                    elif 'train_decoder' in name:
                        param.requires_grad = True
                        unfrozen_param_count += 1
                        module_name = name.split('.')[0]
                        if module_name not in unfrozen_modules:
                            unfrozen_modules.append(module_name)
                        print(f"  [UNFREEZE] {name} (train_decoder)")
                    # evolve/snake 관련 모듈 unfreeze
                    elif any(evolve_name in name.lower() for evolve_name in ['evolve', 'snake', 'evolution', 'ccp']):
                        param.requires_grad = True
                        unfrozen_param_count += 1
                        module_name = name.split('.')[0]
                        if module_name not in unfrozen_modules:
                            unfrozen_modules.append(module_name)
                        print(f"  [UNFREEZE] {name} (evolve/ccp)")
                
                print(f"[STAGE 2] Total unfrozen parameters: {unfrozen_param_count}")
                
                # ✅ head gradient 계산 검증을 위한 추가 정보
                print(f"\n[STAGE 2] Head Gradient Verification:")
                for head_name, trained_in_s1 in [('ct_hm', stage1_trained_ct_hm), ('wh', stage1_trained_wh)]:
                    head_status = f"FROZEN (from Stage 1)" if trained_in_s1 else "TRAINABLE (Stage 2 only)"
                    print(f"  {head_name.upper()} HEAD ({head_status}):")
                    for name, param in net.named_parameters():
                        if head_name in name:
                            print(f"    {name}: requires_grad={param.requires_grad}, shape={param.shape}")
                            # 파라미터 변화 추적을 위한 초기값 저장 (옵션)
                            if not hasattr(param, '_initial_value'):
                                param._initial_value = param.data.clone()
                            print(f"      initial_mean: {param._initial_value.mean().item():.6f}")
                            print(f"      current_mean: {param.data.mean().item():.6f}")
                            if trained_in_s1:
                                print(f"      -> {head_name.upper()} was trained in Stage 1, now frozen")
                
                set_bn_eval_if_frozen(net)
                
                for module_name in unfrozen_modules:
                    print(f"  └ {module_name}: unfrozen")
                        
            else:
                # 기존 방식: 전부 학습
                print(f"[STAGE 2] {task_name}: Training all modules")
                unfreeze_all(net)
        else:
            # 다른 task들은 기존 방식 유지
            unfreeze_all(net)

        # 2) evolve 루프 복원
        cfg.model.evolve_iters = int(getattr(cfg.model, 'evolve_iters_full', cfg.model.evolve_iters or 1) or 1)

        # 3) refine 복원(선택)
        # cfg.model.use_refine_pixel = original_value

        # 4) loss weight 복원(원래 cfg 값으로)
        # Stage 1에서 다른 loss들이 0으로 설정되었으므로 Stage 2에서 복원
        if 'ccp' in cfg.commen.task:
            print("[STAGE 2] Restoring loss weights from config")
            original_weights = dict(cfg.train.weight_dict)
            current_weights = dict(net_wrapper.weight_dict)
            
            # Stage 2에서 필요한 loss들 복원
            # Stage 1에서 학습된 loss는 유지 (pixel, 옵션에 따라 box_ct, init)
            stage1_trained_ct_hm = getattr(cfg.train, 'stage1_train_ct_hm', False)
            stage1_trained_wh = getattr(cfg.train, 'stage1_train_wh', False)
            
            stage1_loss_keys = ['pixel']  # pixel은 항상 Stage 1에서 학습
            if stage1_trained_ct_hm:
                stage1_loss_keys.append('box_ct')
            if stage1_trained_wh:
                stage1_loss_keys.append('init')
            
            for key, value in original_weights.items():
                # Stage 1에서 설정된 loss는 유지
                if not any(s1_key in key for s1_key in stage1_loss_keys):
                    if current_weights.get(key, None) != value:
                        net_wrapper.weight_dict[key] = value
                        print(f"  [RESTORE] {key}: {current_weights.get(key, 'None')} -> {value}")
            
            print(f"[STAGE 2] Final loss weights: {dict(net_wrapper.weight_dict)}")

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # freeze 상태에서 BN 통계 고정
    set_bn_eval_if_frozen(net)

def freeze_all(m):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_all(m):
    for p in m.parameters():
        p.requires_grad = True

def set_bn_eval_if_frozen(m):
    import torch.nn as nn
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            # 모듈 내 파라미터가 모두 freeze면 eval
            if all(p.requires_grad is False for p in mod.parameters(recurse=False)):
                mod.eval()

def enable_pixel_head_only(net, cfg=None):
    """
    Stage 1: pixel head + backbone 학습, 나머지 freeze
    - backbone (dla 등) 학습 가능
    - pixel head 학습 가능  
    - refine 등 다른 모듈은 freeze
    - ct_hm head는 cfg.train.stage1_train_ct_hm 옵션에 따라 결정
    - wh head는 cfg.train.stage1_train_wh 옵션에 따라 결정
    """
    freeze_all(net)
    unfrozen_count = 0
    pixel_params = 0
    backbone_params = 0
    ct_hm_params = 0
    wh_params = 0
    
    # head 학습 여부 확인
    train_ct_hm = getattr(cfg.train, 'stage1_train_ct_hm', False) if cfg else False
    train_wh = getattr(cfg.train, 'stage1_train_wh', False) if cfg else False
    
    for name, p in net.named_parameters():
        # pixel head 활성화 (refine 제외)
        if ('pixel' in name) and ('refine' not in name):
            p.requires_grad = True
            unfrozen_count += 1
            pixel_params += 1
            print(f"  [STAGE1 UNFREEZE] {name} (pixel_head)")
        # ct_hm head 활성화 (옵션에 따라)
        elif train_ct_hm and 'ct_hm' in name:
            p.requires_grad = True
            unfrozen_count += 1
            ct_hm_params += 1
            print(f"  [STAGE1 UNFREEZE] {name} (ct_hm_head)")
        # wh head 활성화 (옵션에 따라)
        elif train_wh and 'wh' in name:
            p.requires_grad = True
            unfrozen_count += 1
            wh_params += 1
            print(f"  [STAGE1 UNFREEZE] {name} (wh_head)")
        # backbone 활성화 (dla, resnet 등)
        elif 'dla' in name.lower() or 'backbone' in name.lower() or 'base_layer' in name:
            p.requires_grad = True
            unfrozen_count += 1
            backbone_params += 1
            print(f"  [STAGE1 UNFREEZE] {name} (backbone)")
            
    # 파라미터 수 출력
    param_summary = [f"Pixel head params: {pixel_params}"]
    if train_ct_hm:
        param_summary.append(f"CT_HM head params: {ct_hm_params}")
    if train_wh:
        param_summary.append(f"WH head params: {wh_params}")
    param_summary.append(f"Backbone params: {backbone_params}")
    
    print(f"[STAGE 1] {', '.join(param_summary)}")
    print(f"[STAGE 1] Total unfrozen parameters: {unfrozen_count}")
    
    # 다른 head들이 정말 freeze되었는지 확인
    frozen_heads = 0
    for name, p in net.named_parameters():
        # ct_hm head는 train_ct_hm 옵션에 따라 다름
        if 'ct_hm' in name and not p.requires_grad:
            if not train_ct_hm:  # ct_hm이 freeze되어야 하는 경우에만 출력
                frozen_heads += 1
                print(f"  [STAGE1 FROZEN] {name} (ct_hm_head)")
        # wh head는 train_wh 옵션에 따라 다름
        elif 'wh' in name and not p.requires_grad:
            if not train_wh:  # wh이 freeze되어야 하는 경우에만 출력
                frozen_heads += 1
                print(f"  [STAGE1 FROZEN] {name} (wh_head)")
    print(f"[STAGE 1] Frozen head parameters: {frozen_heads}")
    
    set_bn_eval_if_frozen(net)