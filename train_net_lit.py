import sys, os, shutil, argparse, importlib, glob, random, numpy as np, cv2
import torch
import pytorch_lightning as pl

# Anomaly detection disabled for performance
# torch.autograd.set_detect_anomaly(True)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy


class CustomEarlyStopping(EarlyStopping):
    """
    EarlyStopping with min_epochs support for older PyTorch Lightning versions
    """
    def __init__(self, *args, min_epochs=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs
        self.min_epochs_reached = False  # 추가: min_epochs 도달 여부 추적
        
    def _should_skip_check(self, trainer):
        # 최소 에포크에 도달하지 않았으면 early stopping 체크를 건너뜀
        if trainer.current_epoch < self.min_epochs:
            return True
        
        # min_epochs에 처음 도달했을 때만 동기화 (한 번만 실행)
        if trainer.current_epoch == self.min_epochs and not self.min_epochs_reached:
            self.min_epochs_reached = True
            
            # ModelCheckpoint에서 현재까지의 best metric 값을 가져옴
            checkpoint_callback = None
            for callback in trainer.callbacks:
                if hasattr(callback, 'best_model_score') and callback.monitor == self.monitor:
                    checkpoint_callback = callback
                    break
            
            if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None:
                # ModelCheckpoint의 best 값으로 동기화 (Tensor 타입 유지)
                old_best = self.best_score
                best_model_score = checkpoint_callback.best_model_score
                # Tensor 타입을 유지해야 PyTorch Lightning과 호환됨
                if hasattr(best_model_score, 'clone'):
                    self.best_score = best_model_score.clone()
                else:
                    import torch
                    self.best_score = torch.tensor(float(best_model_score))
                self.wait_count = 0  # patience 카운트 리셋
                if trainer.is_global_zero:  # 중복 출력 방지
                    print(f"[EarlyStop] Epoch {self.min_epochs}: Synced with ModelCheckpoint")
                    print(f"[EarlyStop] Old best_score: {old_best} -> New best_score: {self.best_score}")
                    print(f"[EarlyStop] Reset wait_count to 0")
            else:
                if trainer.is_global_zero:  # 중복 출력 방지
                    print(f"[EarlyStop] Epoch {self.min_epochs}: No ModelCheckpoint found, using current best_score: {self.best_score}")
                    print(f"[EarlyStop] Reset wait_count to 0")
                self.wait_count = 0  # patience 카운트만 리셋
            
        return super()._should_skip_check(trainer)
from network import make_network
from train.trainer.make_trainer import make_trainer
from train.optimizer.optimizer import make_optimizer
from train.scheduler.scheduler import make_lr_scheduler
from train.recorder.recorder import make_recorder
from dataset.data_loader import make_data_loader
from dataset.collate_batch import collate_batch
from train.model_utils.utils import load_model, save_model, load_network
from evaluator.make_evaluator import make_evaluator

TRAIN_MESSAGE = ""
TRAIN_BEST_METRIC = 0.

import torch.distributed as dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only

@rank_zero_only
def check_dist():
    print("🔥 DDP Initialized:", dist.is_initialized())
    if dist.is_initialized():
        print("🔥 World Size:", dist.get_world_size())
    else:
        print("🔥 Using DataParallel (DP)")

import psutil, os, torch
proc = psutil.Process(os.getpid())

def _memlog(tag=""):
    rss = proc.memory_info().rss/1024**3
    cuda = torch.cuda.max_memory_allocated()/1024**3
    print(f"[MEM] {tag} | RSS={rss:.2f} GB | CUDA-peak={cuda:.2f} GB", flush=True)

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file)
    if args.bs != 'None':
        cfg.train.batch_size = int(int(args.bs) / torch.cuda.device_count())
    if args.test_bs is not None:
        cfg.test.batch_size = int(int(args.test_bs) / torch.cuda.device_count())
    if not args.use_dp:
        cfg.train.use_dp = args.use_dp

    cfg.commen.result_dir = f'{cfg.commen.result_dir}/{args.config_file}/{args.exp}'
    cfg.commen.record_dir = f'{cfg.commen.record_dir}/{args.config_file}/{args.exp}'
    cfg.commen.model_dir = f'{cfg.commen.model_dir}/{args.config_file}/{args.exp}'
    os.makedirs(cfg.commen.result_dir, exist_ok=True)
    shutil.copyfile(f'configs/{args.config_file.replace(".", "/")}.py',
                    f'{cfg.commen.result_dir}/{args.config_file.split(".")[-1]}.py')

    if len(cfg.train.save_ontraining) > 0:
        os.makedirs(f"{cfg.commen.result_dir}/OnTraining", exist_ok=True)

    # cfg.commen.gpus = args.device
    if args.epochs > 0:
        cfg.train.epoch = args.epochs
    cfg.train.validate_first = args.validate_first
    if args.num_workers is not None:
        # ✅ train과 test(validation) 모두의 num_workers를 CLI 인자로 업데이트합니다.
        cfg.train.num_workers = args.num_workers
        cfg.test.num_workers = args.num_workers

    cfg.commen.seed = args.seed
    cfg.commen.deterministic_mode = args.deterministic
    return cfg

def _convert_sync_to_normal_batchnorm_global(module):
    """SyncBatchNorm을 일반 BatchNorm으로 변환 (전역 함수)"""
    import torch.nn as nn
    
    converted_count = 0
    
    def _convert_recursive(mod, path=""):
        nonlocal converted_count
        
        # 현재 모듈이 SyncBatchNorm인지 확인
        if isinstance(mod, torch.nn.SyncBatchNorm):
            print(f"[CONVERT] Found SyncBatchNorm at {path}")
            # SyncBatchNorm을 BatchNorm2d로 변환
            new_module = nn.BatchNorm2d(
                mod.num_features,
                eps=mod.eps,
                momentum=mod.momentum,
                affine=mod.affine,
                track_running_stats=mod.track_running_stats
            )
            if mod.affine:
                with torch.no_grad():
                    new_module.weight = mod.weight
                    new_module.bias = mod.bias
            new_module.running_mean = mod.running_mean
            new_module.running_var = mod.running_var
            new_module.num_batches_tracked = mod.num_batches_tracked
            if hasattr(mod, "qconfig"):
                new_module.qconfig = mod.qconfig
            converted_count += 1
            print(f"[CONVERT] Converted {path} to BatchNorm2d")
            return new_module
        
        # 하위 모듈들을 재귀적으로 변환
        for name, child in mod.named_children():
            new_child = _convert_recursive(child, f"{path}.{name}" if path else name)
            mod.add_module(name, new_child)
        
        return mod
    
    result = _convert_recursive(module)
    print(f"[CONVERT] Total converted SyncBatchNorm modules: {converted_count}")
    return result

class LightningWrapper(pl.LightningModule):
    def __init__(self, network, cfg, network_t=None, use_sync_batchnorm=True):
        super().__init__()
        if torch.cuda.device_count() > 1 and use_sync_batchnorm:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
        self.network = network
        self.cfg = cfg
        self.network_t = network_t
        # self.engine = make_trainer(network, cfg, network_t=network_t)  # 🔥 Trainer 객체 생성
        self.evaluator = make_evaluator(cfg)
        self.recorder = make_recorder(cfg.commen.record_dir)
        from train.trainer.make_trainer import _wrapper_factory as NetworkWrapper
        self.loss_module = NetworkWrapper(self.network, cfg)
        
        # ✅ Pixel task용 evaluator 추가
        self.is_pixel_task = hasattr(cfg.commen, 'task') and cfg.commen.task == 'pixel'
        if self.is_pixel_task:
            from train.trainer.utils import SegmentationMetrics
            self.pixel_evaluator = SegmentationMetrics()
    
    def _convert_sync_to_normal_batchnorm(self, module):
        """SyncBatchNorm을 일반 BatchNorm으로 변환"""
        import torch.nn as nn
        
        module_output = module
        if isinstance(module, torch.nn.SyncBatchNorm):
            # SyncBatchNorm을 BatchNorm2d로 변환
            module_output = nn.BatchNorm2d(
                module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
                
        for name, child in module.named_children():
            module_output.add_module(name, self._convert_sync_to_normal_batchnorm(child))
        del module
        return module_output

    def on_fit_start(self):
        t = self.trainer
        if t.is_global_zero:
            print("[CFG] val_check_interval =", t.val_check_interval, type(t.val_check_interval))
            print("[CFG] check_val_every_n_epoch =", t.check_val_every_n_epoch)
            print("[CFG] limit_train_batches =", t.limit_train_batches, type(t.limit_train_batches))
            print("[CFG] limit_val_batches   =", t.limit_val_batches, type(t.limit_val_batches))
            print("[CFG] num_sanity_val_steps=", t.num_sanity_val_steps)
            print("[CFG] reload_dl_every_n_ep=", getattr(t, "reload_dataloaders_every_n_epochs", None))
            # DDP 디버깅 정보
            print(f"[DDP DEBUG] Strategy: {type(t.strategy).__name__}")
            print(f"[DDP DEBUG] World Size: {t.world_size}")
            print(f"[DDP DEBUG] Global Rank: {t.global_rank}")
            print(f"[DDP DEBUG] Local Rank: {t.local_rank}")
            print(f"[DDP DEBUG] Device count: {torch.cuda.device_count()}")

    def forward(self, batch):
        if ('meta' not in batch) or (batch['meta'] is None):
            batch['meta'] = {'mode': ['unknown']}
        elif not isinstance(batch['meta'], dict):
            batch['meta'] = {'mode': batch['meta']}
        x = batch['inp']
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # print(f"inp : {x.shape}")
        # print(f"batch['ct_hm']: {batch['ct_hm'].shape}")
        return self.network(x, batch=batch)

    def training_step(self, batch, batch_idx):
        # 텐서만 GPU로 올리기
        batch = {
            k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        # print(f"after pre-proces : [TRAIN STEP BEFORE NETWORK] meta={batch.get('meta')}")

        # 🔥 여기서 epoch 추가
        batch['epoch'] = self.current_epoch

        # ✅ NetworkWrapper 호출 (loss 계산 포함)
        output = self.forward(batch)
        
        # 모든 태스크: loss_module에서 loss 계산
        output, loss, loss_stats, _ = self.loss_module.compute_loss(output, batch)
        

        if not torch.isfinite(loss):
            print(f"[❗LOSS ERROR] batch_idx={batch_idx}, loss={loss}", flush=True)
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    print(
                        f"  └ output[{k}]: shape={v.shape}, NaN={torch.isnan(v).any().item()}, Inf={torch.isinf(v).any().item()}",
                        flush=True)

        # === ✅ 로그: 모든 loss를 progress bar에 표시 ===
        # 메인 총합 로스
        self.log('train/loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # 컴포넌트 로스들
        for k, v in loss_stats.items():
            if k == 'loss':
                # 위에서 train/loss로 이미 로깅했으므로 중복 회피
                continue
            # tensor로 보장
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, device=loss.device, dtype=loss.dtype)
            # 모든 loss 항목을 progress bar에 올림
            self.log(f'train/{k}', v,
                     on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_start(self):
        # Stage1 픽셀 IoU 누적기 초기화
        if int(self.cfg.train.stage) == 1:
            self._pix_inter = 0
            self._pix_union = 0
        # ✅ Pixel task용 evaluator 초기화
        if self.is_pixel_task:
            self.pixel_evaluator.reset_results()

    def validation_step(self, batch, batch_idx):
        # 텐서만 GPU로 (한 번만)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device=self.device, non_blocking=True)

        # 🔥 validation에서도 epoch 추가
        batch['epoch'] = self.current_epoch

        # ✅ NetworkWrapper 호출 (validation에서도 loss 계산 가능)
        with torch.no_grad():
            output = self.forward(batch)

        if int(self.cfg.train.stage) == 1:
            with torch.no_grad():
                # pixel map: (B, C/H, H, W)
                pix = output['pixel'][-1] if isinstance(output['pixel'], list) else output['pixel']
                if pix.shape[1] == 1:
                    prob = torch.sigmoid(pix)  # (B,1,H,W)
                    th = float(getattr(self.cfg.test, 'pixel_th', 0.5))
                    pred = (prob >= th).to(torch.bool)
                else:
                    pred = (torch.argmax(pix, dim=1, keepdim=True) != 0)  # (B,1,H,W) fg!=bg

                gt = (batch['pixel_gt'].unsqueeze(1).to(pred.device) > 0)
                # 해상도 맞추기
                if gt.shape[-2:] != pred.shape[-2:]:
                    gt = torch.nn.functional.interpolate(gt.float(), size=pred.shape[-2:], mode='nearest').to(torch.bool)

                inter = (pred & gt).sum().item()
                union = (pred | gt).sum().item()
                self._pix_inter += inter
                self._pix_union += union
        else:
            # ✅ Pixel 태스크: pixel_evaluator 사용
            if self.is_pixel_task:
                import torch.nn.functional as F
                pix = output['pixel'][-1] if isinstance(output['pixel'], list) else output['pixel']
                pixel_gt = F.interpolate(batch['pixel_gt'].unsqueeze(1).float(), 
                                        size=(pix.size(-2), pix.size(-1)), mode='nearest').squeeze(1)
                # pixel_gt를 0-1로 정규화
                pixel_gt = (pixel_gt > 0.5).long()
                self.pixel_evaluator.stack_results(pixel_gt, pix)
            # ✅ 다른 태스크: COCO evaluator 사용
            elif self.evaluator is not None:
                self.evaluator.evaluate(output, batch)

            # 메모리 즉시 해제
            del output, batch
            # 필요 시 주기적으로만 켜두세요 (느려질 수 있음)
            # if batch_idx % 50 == 0:
            #     torch.cuda.empty_cache()

            # ❗ 절대 per-step 출력 반환하지 않기 (Lightning 1.6.x가 쌓음)
        return None

    def validation_epoch_end(self, outputs):
        # outputs는 무시 (우리는 step에서 아무것도 안 돌려줌)
        # torch.cuda.synchronize()는 꼭 필요하진 않지만, 남겨도 무해
        # torch.cuda.synchronize()
        if int(self.cfg.train.stage) == 1:
            # 픽셀 IoU를 val_metric으로 로그
            iou = float(self._pix_inter) / float(self._pix_union + 1e-6)
            print(f"[VAL] Epoch {self.current_epoch}: Stage1 Pixel IoU = {iou:.4f}")
            self.log("val_metric", iou, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        else:
            # ✅ Pixel 태스크: train.val_metric에 따라 metric 선택
            if self.is_pixel_task:
                pixel_acc, dice, precision, recall = self.pixel_evaluator()
                val_metric_name = getattr(self.cfg.train, 'val_metric', 'dice')
                if val_metric_name == 'acc':
                    val_metric = pixel_acc
                elif val_metric_name == 'dice':
                    val_metric = dice
                elif val_metric_name == 'f1':
                    val_metric = 2 * (precision * recall) / (precision + recall + 1e-6)
                else:
                    val_metric = dice  # default
                print(f"[VAL] Epoch {self.current_epoch}: Pixel {val_metric_name} = {val_metric:.4f}")
                self.log("val_metric", val_metric, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
                self.pixel_evaluator.reset_results()
            # ✅ 다른 태스크: COCO AP 사용
            elif self.evaluator is not None:
                result, _ = self.evaluator.summarize(print_out=self.trainer.is_global_zero)
                main_metric = float(next(iter(result.values()), 0.0))
                print(f"[VAL] Epoch {self.current_epoch}: Stage2 AP = {main_metric:.4f}")
                self.log("val_metric", main_metric, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
                # ✅ 누적 비우기 (가장 중요)
                self.evaluator.reset()

        # 가비지 회수
        import gc, torch as _torch
        gc.collect(); _torch.cuda.empty_cache()

        # Lightning은 여기 return 값 사용 안 함. 반환 없어도 됨.
        # return main_metric

    def configure_optimizers(self):
        optimizer = make_optimizer(self.network, self.cfg)
        scheduler = make_lr_scheduler(optimizer, self.cfg)
        if self.cfg.train.optimizer['scheduler'] == 'ReduceLROnPlateau':
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_metric"}}
        else:
            return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        if isinstance(self.trainer.train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)
        if self.trainer.is_global_zero:
            print(f"[E{self.current_epoch}] on_train_epoch_start", flush=True)
            

    def on_validation_epoch_start(self):
        for loader in self.trainer.val_dataloaders:
            if isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
                loader.sampler.set_epoch(self.current_epoch)
        if self.trainer.is_global_zero:
            print(f"[E{self.current_epoch}] on_validation_epoch_start", flush=True)

    def on_test_epoch_start(self):
        for loader in self.trainer.test_dataloaders:
            if isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
                loader.sampler.set_epoch(self.current_epoch)

    @rank_zero_only
    def on_train_end(self):
        last_ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, "last.ckpt")
        if os.path.exists(last_ckpt_path):
            os.remove(last_ckpt_path)
            print(f"[INFO] Removed last.ckpt after training: {last_ckpt_path}")

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % 20 == 0:
            _memlog(f"val step {batch_idx}")

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            print(f"[E{self.current_epoch}] on_train_epoch_end", flush=True)
            

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            print(f"[E{self.current_epoch}] on_validation_epoch_end", flush=True)


def get_latest_ckpt_path(ckpt_dir):
    """
    ckpt_dir 하위에서 가장 최근에 수정된 .ckpt 파일 경로를 반환.
    .ckpt 파일이 없으면 None 반환
    """
    ckpt_file = None
    if ckpt_dir is not None:
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if ckpt_files:
            # 수정 시간 기준 정렬 (내림차순)
            ckpt_files.sort(key=os.path.getmtime, reverse=True)
            ckpt_file = ckpt_files[0]
    return ckpt_file

def main(args):
    # ✅ DDP 환경변수 설정 (deadlock 방지)
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # ✅ NCCL 설정 (교착상태 방지)
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30분
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    
    # ✅ DataLoader와 OpenCV의 충돌로 인한 교착 상태(deadlock)를 방지합니다.
    cv2.setNumThreads(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    pl.seed_everything(args.seed, workers=True)  # ✅ 학습 reproducibility 보장
    cfg = get_cfg(args)
    
    if args.deterministic in ("full", "not_pl"):
        task_name = cfg.commen.task.split('+')[0]
        
        # 모든 태스크에서 cross_entropy 등의 비-deterministic 연산 때문에 warn_only 옵션 사용
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            print(f"Deterministic algorithms enabled with warn_only=True for task: {task_name}")
        except TypeError:
            print(f"Warning: PyTorch version does not support warn_only. Disabling deterministic for {task_name}.")
            torch.use_deterministic_algorithms(False)
            
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Teacher model
    if 'kd' in cfg.commen.task:
        cfg_t = importlib.import_module('configs.' + cfg.model.teacher_cfg_name)
        network_t = make_network.get_network(cfg_t).cuda()
    else:
        network_t = None

    network = make_network.get_network(cfg).cuda()

    stage = int(cfg.train.stage)
    
    # ✅ Stage 2 학습 시 Stage 1 모델 로드 (selective loading만 사용)
    if stage == 2:
        best_s1_path = os.path.join(cfg.commen.model_dir, "best_s1.pth")
        if os.path.exists(best_s1_path):
            print(f"[STAGE 2] Loading Stage 1 model from: {best_s1_path}")
            # selective loading 설정 확인
            use_selective = getattr(cfg.train, 'use_selective_s1_loading', True)
            if use_selective:
                try:
                    from train.model_utils.utils import load_stage1_modules_selective
                    if load_stage1_modules_selective(network, best_s1_path, cfg):
                        print(f"[STAGE 2] Selectively loaded Stage 1 modules for fine-tuning")
                    else:
                        print(f"[STAGE 2] Selective loading failed - training from scratch")
                except Exception as e:
                    print(f"[STAGE 2] Selective loading error: {e}")
                    print(f"[STAGE 2] Starting Stage 2 training from scratch")
            else:
                print(f"[STAGE 2] Selective loading disabled by config - training from scratch")
        else:
            print(f"[STAGE 2] Warning: Stage 1 model not found at {best_s1_path}")
            print(f"[STAGE 2] Starting Stage 2 training from scratch")

    # SyncBatchNorm 사용 안함일 때 모든 SyncBatchNorm을 먼저 제거
    print("="*80)
    print(f"🔥 [SYNC_BATCHNORM] args.sync_batchnorm = {args.sync_batchnorm}")
    print("="*80)
    if not args.sync_batchnorm:
        print("[INFO] Converting all SyncBatchNorm to BatchNorm2d...")
        
        # 1차: PyTorch 내장 함수로 변환 시도
        try:
            network = torch.nn.utils.convert_sync_batchnorm.convert_sync_batchnorm(network, process_group=None)
            if network_t is not None:
                network_t = torch.nn.utils.convert_sync_batchnorm.convert_sync_batchnorm(network_t, process_group=None)
            print("[INFO] Used PyTorch built-in convert_sync_batchnorm")
        except Exception as e:
            print(f"[WARNING] PyTorch built-in conversion failed: {e}")
            print("[INFO] Using custom conversion function...")
            
        # 2차: 커스텀 함수로 추가 변환 (혹시 놓친 것들)
        network = _convert_sync_to_normal_batchnorm_global(network)
        if network_t is not None:
            network_t = _convert_sync_to_normal_batchnorm_global(network_t)
    
    model = LightningWrapper(network, cfg, network_t=network_t, use_sync_batchnorm=args.sync_batchnorm)
    
    # ✅ CUDA Warmup for 3x3 feature extraction (reduces sanity check time)
    if cfg.model.use_3x3_feature and torch.cuda.is_available():
        try:
            from warmup_cuda import warmup_cuda_kernels
            warmup_cuda_kernels(device='cuda', 
                              feature_dim=64,  # Adjust based on your feature dimension
                              num_vertices=cfg.commen.init_points_per_poly,
                              h=104, w=104)  # Feature map size for 416x416 input with stride 4
        except Exception as e:
            print(f"CUDA warmup failed (non-critical): {e}")

    # ✅ Stage별 체크포인트 파일명 설정
    if stage == 1:
        checkpoint_filename = "best_s1"
        save_filename = "{epoch}-{val_metric:.4f}-s1"
    else:  # stage == 2 or default
        checkpoint_filename = "best"
        save_filename = "{epoch}-{val_metric:.4f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.commen.model_dir,
        filename=save_filename,
        save_top_k=1,
        monitor="val_metric",
        mode="max" if cfg.train.best_metric_crit == "max" else "min",
        save_last=True,  # ✅ 마지막 에폭 저장
    )

    early_stop_callback = CustomEarlyStopping(
        monitor="val_metric",
        patience=cfg.train.earlystop,
        mode="max" if cfg.train.best_metric_crit == "max" else "min",
        min_epochs=getattr(cfg.train, 'min_epochs_for_earlystop', 50),  # 최소 에포크 수 후에만 early stopping 활성화
        check_on_train_epoch_end=False,  # ✅ 핵심: 검증 끝난 후에만 체크
    )


    num_gpus = torch.cuda.device_count()
    if args.gpu_strategy == 'ddp' and num_gpus > 1:
        # ✅ DDP 교착상태 방지를 위한 추가 설정
        from datetime import timedelta
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(minutes=2),  # 2분 타임아웃
            process_group_backend="nccl",
        )
        print(f"✅ Using DDP strategy with {num_gpus} GPUs")
    elif args.gpu_strategy == 'dp' and num_gpus > 1:
        strategy = "dp"
        print(f"✅ Using DataParallel (DP) with {num_gpus} GPUs")
    else:
        strategy = "auto"  # single GPU면 자동
        print(f"✅ Using single GPU strategy")

    trainer = pl.Trainer(
        max_epochs=cfg.train.epoch,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=cfg.commen.result_dir,
        log_every_n_steps=10,
        precision=args.precision,
        profiler="simple",
        deterministic=True if args.deterministic == 'full' else False,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=1.0,  # ✅ 에폭 끝에서만 검증
        check_val_every_n_epoch=cfg.test.check_val_every_n_epoch,  # 매 에폭
        num_sanity_val_steps=2,
        limit_train_batches=1.0,  # ← float!
        limit_val_batches=1.0,  # ← float!
    )
    # print(trainer.strategy)
    # print(type(trainer.strategy))
    #
    # print("gpus:", trainer.gpus)
    # print("devices:", trainer.devices)
    # print("num_processes:", trainer.num_processes)
    # print("num_nodes:", trainer.num_nodes)

    print("🔍 [DEBUG] Creating data loaders...")
    train_loader, val_loader = make_data_loader(cfg=cfg, val_split=cfg.data.val_split)
    print(f"🔍 [DEBUG] Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")
    # ✅ collate_fn 지정
    # train_loader.collate_fn = collate_batch
    # val_loader.collate_fn = collate_batch
    # print(f"[DEBUG] train_loader.collate_fn = {train_loader.collate_fn}")
    # print(f"[DEBUG] val_loader.collate_fn = {val_loader.collate_fn}")

    # sample = val_loader.dataset[0]
    # print("[DEBUG] Sample keys:", sample.keys())

    # checkpoint 경로 결정
    ckpt_path = get_latest_ckpt_path(cfg.commen.model_dir)

    print("🚀 [DEBUG] Starting training...")
    print(f"🚀 [DEBUG] Checkpoint path: {ckpt_path}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    print("✅ [DEBUG] Training completed successfully")

    best_path = checkpoint_callback.best_model_path
    if best_path:
        import shutil
        # ✅ Stage별로 다른 파일명으로 저장
        if stage == 1:
            target_filename = "best_s1.pth"
        else:
            target_filename = "best.pth"
        
        target_path = os.path.join(cfg.commen.model_dir, target_filename)
        shutil.copy(best_path, target_path)
        print(f"✅ [STAGE {stage}] Best model copied to: {target_path}")

if __name__ == "__main__":
    import signal, sys, torch.distributed as dist
    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()

    def handler(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--exp", default="None")
    parser.add_argument("--checkpoint", default="None")
    parser.add_argument("--type", default="continue")
    parser.add_argument("--bs", default="None")
    parser.add_argument("--test_bs", default=None)
    # parser.add_argument("--device", default=0, nargs="+", type=int, help='device idx')
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--validate_first", default=True, type=bool, choices=[True, False])
    parser.add_argument("--use_dp", default=True, type=bool, choices=[True, False])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--precision", default=32, choices=[32,16,64], type=int)
    parser.add_argument("--gpu_strategy", default="dp")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="Accumulates grads every k batches.")
    parser.add_argument("--deterministic", default="full", choices=["full","not_pl","never"])
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument("--sync_batchnorm", default=True, type=str2bool, help="Use SyncBatchNorm in multi-GPU training")

    args = parser.parse_args()

    main(args)
