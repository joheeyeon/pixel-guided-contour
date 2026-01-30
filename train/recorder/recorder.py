from collections import deque, defaultdict
import torch
from tensorboardX import SummaryWriter
import os
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


# class Recorder(object):
#     def __init__(self, record_dir, task='e2ec', use_smoothing=False): #rev : 24-09-08
#         log_dir = record_dir
#         self.writer = None
#         if (not dist.is_initialized()) or (dist.get_rank() == 0):
#             os.makedirs(record_dir, exist_ok=True)
#             self.writer = SummaryWriter(log_dir=log_dir)
#         self.use_smoothing = use_smoothing
#
#         # scalars
#         self.epoch = 0
#         self.step = 0
#         if self.use_smoothing:
#             self.loss_stats = defaultdict(SmoothedValue)
#         else:
#             self.loss_stats = {}
#
#         self.batch_time = SmoothedValue()
#         self.data_time = SmoothedValue()
#
#         # images
#         self.image_stats = defaultdict(object)
#         if 'process_' + task in globals():
#             self.processor = globals()['process_' + task]
#         else:
#             self.processor = None
#
#     # def update_loss_stats(self, loss_dict):
#     #     for k, v in loss_dict.items():
#     #         try:
#     #             self.loss_stats[k].update(v.detach().cpu())
#     #         except:
#     #             self.loss_stats[k] = v.detach().cpu()
#     def update_loss_stats(self, loss_dict):
#         for k, v in loss_dict.items():
#             try:
#                 if isinstance(v, torch.Tensor):
#                     self.loss_stats[k].update(v.detach().cpu())
#                 else:  # 이미 float이면 바로 업데이트
#                     self.loss_stats[k].update(v)
#             except KeyError:
#                 if isinstance(v, torch.Tensor):
#                     self.loss_stats[k] = SmoothedValue()
#                     self.loss_stats[k].update(v.detach().cpu())
#                 else:
#                     self.loss_stats[k] = SmoothedValue()
#                     self.loss_stats[k].update(v)
#
#     def update_image_stats(self, image_stats):
#         if self.processor is None:
#             return
#         image_stats = self.processor(image_stats)
#         for k, v in image_stats.items():
#             self.image_stats[k] = v.detach().cpu()
#
#     def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
#         pattern = prefix + '/{}'
#         step = step if step >= 0 else self.step
#         loss_stats = loss_stats if loss_stats else self.loss_stats
#
#         if self.writer:
#             for k, v in loss_stats.items():
#                 if isinstance(v, SmoothedValue):
#                     self.writer.add_scalar(pattern.format(k), v.median, step)
#                 else:
#                     try:
#                         self.writer.add_scalar(pattern.format(k), v.item(), step)
#                     except:
#                         self.writer.add_scalar(pattern.format(k), v, step)
#
#         if self.processor is None:
#             return
#         image_stats = self.processor(image_stats) if image_stats else self.image_stats
#         if self.writer:
#             for k, v in image_stats.items():
#                 self.writer.add_image(pattern.format(k), v, step)
#
#     def state_dict(self):
#         return {'step': self.step}
#
#     def load_state_dict(self, state):
#         self.step = state['step']
#
#     def __str__(self):
#         loss_state = []
#         for k, v in self.loss_stats.items():
#             if isinstance(v, SmoothedValue):
#                 loss_state.append('{}: {:.4f}'.format(k, v.avg))
#             else:
#                 try:
#                     loss_state.append('{}: {:.4f}'.format(k, v.item()))
#                 except:
#                     loss_state.append('{}: {:.4f}'.format(k, float(v)))
#         loss_state = '  '.join(loss_state)
#         return '  '.join(['epoch: {}', 'step: {}', '{}']).format(self.epoch, self.step, loss_state)

class Recorder(object):
    def __init__(self, record_dir, task='e2ec', use_smoothing=False):
        self.use_smoothing = use_smoothing
        self.epoch = 0
        self.step = 0

        # loss stats
        if self.use_smoothing:
            self.loss_stats = defaultdict(SmoothedValue)
        else:
            self.loss_stats = {}

        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # image stats
        self.image_stats = defaultdict(object)

        # only rank 0 creates SummaryWriter
        self.is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        self.writer = None
        if self.is_main_process:
            os.makedirs(record_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=record_dir)

        # optional image processing
        self.processor = globals().get('process_' + task, None)

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            try:
                if isinstance(v, torch.Tensor):
                    self.loss_stats[k].update(v.detach().cpu())
                else:
                    self.loss_stats[k].update(v)
            except KeyError:
                self.loss_stats[k] = SmoothedValue()
                if isinstance(v, torch.Tensor):
                    self.loss_stats[k].update(v.detach().cpu())
                else:
                    self.loss_stats[k].update(v)

    def update_image_stats(self, image_stats):
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        if not self.is_main_process:
            return

        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                try:
                    self.writer.add_scalar(pattern.format(k), v.item(), step)
                except:
                    self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            return
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        return {'step': self.step}

    def load_state_dict(self, state):
        self.step = state['step']

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            if isinstance(v, SmoothedValue):
                loss_state.append(f'{k}: {v.avg:.4f}')
            else:
                try:
                    loss_state.append(f'{k}: {v.item():.4f}')
                except:
                    loss_state.append(f'{k}: {float(v):.4f}')
        return '  '.join(['epoch: {}', 'step: {}', '{}']).format(self.epoch, self.step, '  '.join(loss_state))


def make_recorder(record_dir):
    return Recorder(record_dir=record_dir)

