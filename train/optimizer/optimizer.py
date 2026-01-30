import torch


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def make_optimizer(net, cfg):
    with_rasterize_net = cfg.model.with_rasterize_net
    opt_cfg = cfg.train.optimizer
    lr = opt_cfg['lr']
    weight_decay = opt_cfg['weight_decay']
    
    # temperature parameter들을 위한 별도 설정
    temperature_lr = opt_cfg.get('temperature_lr', lr)  # default는 일반 lr과 동일
    temperature_weight_decay = opt_cfg.get('temperature_weight_decay', weight_decay)  # default는 일반 weight_decay와 동일
    
    params = []
    temperature_params = []
    n_total = n_trainable = n_temperature = 0
    
    for key, value in net.named_parameters():
        n_total += value.numel()
        if not value.requires_grad:
            continue
        if with_rasterize_net and ('rasterizer' in key):
            continue
        n_trainable += value.numel()
        
        # temperature parameter인지 확인 (trainable_softmax에서 사용되는 temperature)
        if 'temperature' in key and hasattr(cfg.model, 'ccp_deform_pixel_norm') and cfg.model.ccp_deform_pixel_norm == 'trainable_softmax':
            temperature_params.append({"params": [value], "lr": temperature_lr, "weight_decay": temperature_weight_decay})
            n_temperature += value.numel()
            print(f"[OPT] Temperature parameter found: {key} -> lr={temperature_lr}, wd={temperature_weight_decay}")
        else:
            params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

    # temperature parameter group 추가
    if temperature_params:
        params.extend(temperature_params)
        
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found. Did you call apply_stage(...) before make_optimizer()?")

    # Optimizer 생성
    name = opt_cfg.get('name', 'adam')
    if 'adam' in name:
        optimizer = _optimizer_factory[name](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[name](params, lr, momentum=0.9)

    # initial_lr 설정 (scheduler를 위해 필요)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    # (선택) 간단 로그
    print(f"[OPT] stage={int(getattr(cfg.train, 'stage', 2))} lr={lr} wd={weight_decay} | "
          f"params: trainable={n_trainable:,} / total={n_total:,}")
    if n_temperature > 0:
        print(f"[OPT] Temperature params: {n_temperature:,} with separate lr={temperature_lr}, wd={temperature_weight_decay}")

    return optimizer
