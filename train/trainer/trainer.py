import time
import datetime
import torch
import tqdm
from scipy.io import savemat
import numpy as np
from .utils import SegmentationMetrics
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

def print_progress_ontxt(fout, message):
    f = open(fout, "a")
    f.write(f'{message}\n')
    f.close()

def safe_empty_tensor(tensor):
    if tensor.numel() == 0:
        # tensor shape 예: (0, 64, 2) => (1, 64, 2) 이렇게 수정
        shape = list(tensor.shape)
        if shape[0] == 0:
            shape[0] = 1  # batch dim만 1로 변경
        empty_tensor = torch.zeros(
            shape,
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad
        )
        return empty_tensor
    else:
        return tensor

class Trainer(object):
    def __init__(self, network, cfg, network_t=None):
        # print(f"(pre) isinstance(network, torch.nn.DataParallel) : {isinstance(network, torch.nn.DataParallel)}")
        if hasattr(network, 'module'):
            print("Network is wrapped with DataParallel")
        else:
            print("Network is a single model")
        print(f"isinstance(network, torch.nn.DataParallel) : {isinstance(network, torch.nn.DataParallel)}")
        self.network = network
        self.network_t = network_t
        self.fout_train = f'{cfg.commen.record_dir}/train_progress.txt'
        self.fout_val = f'{cfg.commen.record_dir}/val_progress.txt'
        self.cfg = cfg
        self.start_region_loss = False
        self.params_backup = {}
        self.backup_type = self.cfg.train.optimizer['backup_type'] if 'backup_type' in self.cfg.train.optimizer else 'single'
        self.n_save_backup = 10
        self.batch_backup = []
        self.n_save_batch = self.cfg.train.optimizer['n_save_batch'] if 'n_save_batch' in self.cfg.train.optimizer else 0
        self.is_all_simple_pre_epoch = False
        self.add_iou_loss = False
        self.pixel_evaluator = SegmentationMetrics()
        self.scaler = GradScaler()

    def backup_batch(self, batch):
        self.batch_backup.append(batch)
        if len(self.batch_backup) > self.n_save_batch:
            del self.batch_backup[0]

    def backup_parameters(self, model, step):
        if self.backup_type == 'multi':
            self.params_backup[step] = {name: param.clone() for name, param in model.named_parameters()}
            if len(self.params_backup) > self.n_save_backup:
                del self.params_backup[step-self.n_save_backup]
        else:
            self.params_backup = {name: param.clone() for name, param in model.named_parameters()}

    def rollback_parameters(self, model, step):
        # backup_type = self.cfg.train.optimizer['backup_type'] if 'backup_type' in self.cfg.train.optimizer else 'single'
        if self.backup_type == 'multi':
            for name, param in model.named_parameters():
                param.data.copy_(self.params_backup[step][name])
        else:
            for name, param in model.named_parameters():
                param.data.copy_(self.params_backup[name])

    def adjust_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {}

        for k, v in loss_stats.items():
            if isinstance(v, torch.Tensor):
                val = v.detach()
            else:
                val = torch.tensor(v, dtype=torch.float32, device='cuda')

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
                val /= torch.distributed.get_world_size()

            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    reduced_losses[k] = val.item()
                else:
                    reduced_losses[k] = val.mean().item()
            else:
                reduced_losses[k] = float(val)  # float 타입일 수도 있으니 안전 처리
            # reduced_losses[k] = val.item()
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        if self.network_t is not None:
            self.network_t.eval()
        if hasattr(self.network, 'module'):
            self.network.module.net.net_preprocess()
        else:
            self.network.net.net_preprocess()
        end = time.time()
        self.backup_lr = optimizer.param_groups[0]['lr']
        if ('schedule_type' in self.cfg.train.iou_params) and (not self.add_iou_loss):
            if (self.cfg.train.iou_params['schedule_type'] == 'after_simple') and (self.is_all_simple_pre_epoch):
                self.add_iou_loss = True
        is_all_simple = True
        count_all_not_simple = 0
        for iteration, batch in enumerate(data_loader):
            count_not_simple = 0
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            batch = self.to_cuda(batch)
            batch.update({'epoch': epoch})
            if self.n_save_batch > 0:
                self.backup_batch(batch)

            # rev23 : adjust lr on training : 24-09-11
            if 'rollback_gamma' in self.cfg.train.optimizer:
                count_try = 0
                while True:
                    # 0-1. preview
                    if self.n_save_batch > 0:
                        is_py_simple_final = True
                        for preview_batch in self.batch_backup:
                            out_preview, is_py_simple = self.network(preview_batch, mode='preview')
                            if not torch.all(is_py_simple):
                                is_py_simple_final = False
                                break
                    else:
                        out_preview, is_py_simple = self.network(batch, mode='preview')
                        is_py_simple_final = torch.all(is_py_simple)

                    count_try += 1
                    # 0-2. is simple?
                    if is_py_simple_final:
                        # 0-2-1. yes; -> 1
                        dict_ontraining = {'py': out_preview['py_pred'][-1].clone().detach().cpu().numpy(),
                                           'is_py_simple': is_py_simple.clone().detach().cpu().numpy()}
                        savemat(f"{self.cfg.commen.result_dir}/OnTraining/e{epoch}_{iteration}_finalsucess.mat",
                                dict_ontraining)
                        # print(f"final success : {torch.nonzero(is_py_simple.logical_not())}")
                        break
                    elif (count_try > self.n_save_backup):
                        when_end_rollback = self.cfg.train.optimizer['when_end_rollback'] if 'when_end_rollback' in self.cfg.train.optimizer else 'final_update'
                        if when_end_rollback == 'rollback':
                            # print(
                            #     f"before roll back, where is not simple? {torch.nonzero(is_py_simple.logical_not())}")
                            self.rollback_parameters(self.network, recorder.step - count_try)
                            out_preview, is_py_simple = self.network(batch, mode='preview')
                            # print(f"after roll back, where is not simple? {torch.nonzero(is_py_simple.logical_not())}")
                            dict_ontraining = {'py': out_preview['py_pred'][-1].clone().detach().cpu().numpy(),
                                               'is_py_simple': is_py_simple.clone().detach().cpu().numpy()}
                            savemat(f"{self.cfg.commen.result_dir}/OnTraining/e{epoch}_{iteration}_rollback.mat",
                                    dict_ontraining)
                        else:
                            dict_ontraining = {'py': out_preview['py_pred'][-1].clone().detach().cpu().numpy(),
                                               'is_py_simple': is_py_simple.clone().detach().cpu().numpy()}
                            savemat(f"{self.cfg.commen.result_dir}/OnTraining/e{epoch}_{iteration}_finalfail.mat",
                                    dict_ontraining)
                            # print(f"final fail : {torch.nonzero(is_py_simple.logical_not())}")
                        break
                    else:
                        dict_ontraining = {'py': out_preview['py_pred'][-1].clone().detach().cpu().numpy(),
                                           'is_py_simple': is_py_simple.clone().detach().cpu().numpy()}
                        savemat(f"{self.cfg.commen.result_dir}/OnTraining/e{epoch}_{iteration}_{count_try}.mat",
                                dict_ontraining)
                        # 0-2-2. no; roll back params. & lr down ->  0-3
                        self.rollback_parameters(self.network, recorder.step-count_try)
                        current_lr = optimizer.param_groups[0]['lr']
                        new_lr = current_lr * self.cfg.train.optimizer['rollback_gamma']
                        self.adjust_learning_rate(optimizer, new_lr)
                        # print(f"lr adjusted : {current_lr} -> {new_lr}")
                        # 0-3. update params. -> 0-1
                        optimizer.step()
                output, loss, loss_stats, dict_ontraining = self.network(batch)
                loss = loss.mean()

                self.adjust_learning_rate(optimizer, self.backup_lr) #roll back lr
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                # 2. BACK UP : params & lr
                if hasattr(self.network, 'module'):
                    self.backup_parameters(self.network.module, recorder.step)
                else:
                    self.backup_parameters(self.network, recorder.step)
                # self.backup_lr = optimizer.param_groups[0]['lr']
                # 3. params. update
                if dist.is_initialized():
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                # save "on training"
                if len(dict_ontraining) > 0:
                    for key, val in dict_ontraining.items():
                        if len(val) > 0:
                            try:
                                dict_ontraining[key] = np.stack(val)
                            except:
                                max_num = np.array([each_val.shape[0] for each_val in val]).max()
                                new_val = [np.pad(each_val, ((0, max_num - each_val.shape[0]), (0, 0))) for each_val in
                                           val]
                                dict_ontraining[key] = np.stack(new_val)
                    savemat(f"{self.cfg.commen.result_dir}/OnTraining/e{epoch}_{iteration}.mat",
                            dict_ontraining)
            else:
                if self.network_t is not None:
                    output_t = self.network_t(batch['inp'], batch=batch)
                else:
                    output_t = None

                if self.cfg.train.apply_amp:
                    with autocast():
                        output, loss, loss_stats, dict_ontraining = self.network(batch,
                                                                                 mode='default' if not self.add_iou_loss else 'add_iou_loss',
                                                                                 output_t=output_t)
                else:
                    output, loss, loss_stats, dict_ontraining = self.network(batch, mode='default' if not self.add_iou_loss else 'add_iou_loss', output_t=output_t)
                # 🔥 output 안에 있는 list 들까지 안전하게 변환
                for k, v in output.items():
                    if isinstance(v, list):
                        output[k] = [
                            safe_empty_tensor(item) if isinstance(
                                item, torch.Tensor) else item for item in v]
                    elif isinstance(v, torch.Tensor):
                        output[k] = safe_empty_tensor(v)

                if ('schedule_type' in self.cfg.train.iou_params):
                    if (self.cfg.train.iou_params['schedule_type'] == 'after_simple') and (not self.add_iou_loss):
                        if not np.all(dict_ontraining['is_simple']):
                            is_all_simple = False
                            count_not_simple += np.count_nonzero(np.logical_not(dict_ontraining['is_simple']))
                            count_all_not_simple += count_not_simple
                            # print(f"where is not simple? {np.nonzero(np.logical_not(dict_ontraining['is_simple']))}")

                # params. update
                loss = loss.mean()
                optimizer.zero_grad()
                # start = time.time()
                if self.cfg.train.apply_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                    optimizer.step()

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            flag_print = False
            if dist.is_initialized() and dist.get_rank() == 0:
                flag_print = True
            if not dist.is_initialized():
                flag_print = True

            if (iteration % 10 == 0 or iteration == (max_iter - 1)) and flag_print:
                # save "on training"
                if len(self.cfg.train.save_ontraining) > 0:
                    for key, val in dict_ontraining.items():
                        if len(val) > 0:
                            try:
                                dict_ontraining[key] = np.stack(val)
                            except:
                                max_num = np.array([each_val.shape[0] for each_val in val]).max()
                                new_val = [np.pad(each_val, ((0, max_num - each_val.shape[0]), (0, 0))) for each_val in
                                           val]
                                dict_ontraining[key] = np.stack(new_val)

                    savemat(f"{self.cfg.commen.result_dir}/OnTraining/e{epoch}_{iteration}.mat",
                            dict_ontraining)
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)
                print_progress_ontxt(self.fout_train, training_state)
                recorder.record('train')
                if (self.cfg.train.iou_params['schedule_type'] == 'after_simple'):
                    print(f"number of not simple : {count_not_simple}")

        if ('schedule_type' in self.cfg.train.iou_params) and (not self.is_all_simple_pre_epoch):
            if self.cfg.train.iou_params['schedule_type'] == 'after_simple':
                print(f"is all simple? {count_all_not_simple}")
                self.is_all_simple_pre_epoch = is_all_simple

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        flag_print = True
        if dist.is_initialized() and dist.get_rank() != 0:
            flag_print = False

        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        val_metric_sum = 0.
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            batch.update({'epoch': epoch})
            with torch.no_grad():
                if self.cfg.commen.task.split('+')[0] == 'pixel':
                    if self.cfg.train.val_metric == 'loss':
                        output, loss, result, _ = self.network(batch, mode='get_loss')
                        loss = loss.mean()
                        val_out = loss
                    else:
                        output = self.network(batch)
                        pixel_gt = torch.nn.functional.interpolate(batch['pixel_gt'].unsqueeze(1).float(),
                                                 size=(output['pixel'].size(-2), output['pixel'].size(-1)), mode='nearest').squeeze(1)
                        self.pixel_evaluator.stack_results(pixel_gt, output['pixel'])
                else:
                    output = self.network(batch)
                    if self.cfg.train.validate_with_reduction:
                        # BUG FIX: 검증 시 후처리 결과를 append가 아닌 replace하여 평가해야 올바른 모델이 저장됩니다.
                        output['py'][-1] = output['py_reduced']
                    if evaluator is not None and flag_print:
                        evaluator.evaluate(output, batch)

        if self.cfg.commen.task.split('+')[0] != 'pixel':
            if evaluator is not None and flag_print:
                result, _ = evaluator.summarize(print_out=flag_print)
                val_loss_stats.update(result)
                print_progress_ontxt(self.fout_val, f"{'-' * 5} epoch {epoch} {'-' * 5}")
                for k, v in result.items():
                    print_progress_ontxt(self.fout_val, f"{k} : {v}")

                print_progress_ontxt(self.fout_val, f"\n")
            else:
                result = []
        else:
            # val_out = val_metric_sum / len(data_loader)
            pixel_acc, dice, precision, recall = self.pixel_evaluator()
            if self.cfg.train.val_metric == 'acc':
                val_out = pixel_acc
            elif self.cfg.train.val_metric == 'dice':
                val_out = dice
            elif self.cfg.train.val_metric == 'f1':
                val_out = 2 * (precision * recall) / (precision + recall)

        if recorder:
            recorder.record('val', epoch, val_loss_stats)

        if self.cfg.commen.task.split('+')[0] == 'pixel':
            self.pixel_evaluator.reset_results()
            return val_out
        else:
            return result[list(result)[0]] if len(result) > 0 else result