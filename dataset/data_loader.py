import torch
from .collate_batch import collate_batch
from .info import DatasetInfo
import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataset(dataset_name, is_test, cfg, split=None):
    info = DatasetInfo.dataset_info[dataset_name]
    if cfg.data.class_type == 'RasterDataset':
        from .train import base
        dataset = base.RasterDataset(info['anno_file_list'], info['image_dir'], info['split'] if split is None else split, cfg)
    else:
        if is_test:
            from .test import coco, cityscapes, cityscapesCoco, sbd, kitti, pcb, whu
            dataset_dict = {'coco': coco.CocoTestDataset, 'cityscapes': cityscapes.Dataset,
                            'cityscapesCoco': cityscapesCoco.CityscapesCocoTestDataset,
                            'kitti': kitti.KittiTestDataset, 'sbd': sbd.SbdTestDataset,
                            'pcb': pcb.PcbTestDataset, 'whu': whu.WhuTestDataset}
            dataset = dataset_dict[info['name']]
        else:
            from .train import coco, cityscapes, cityscapesCoco, sbd, kitti, pcb, whu
            dataset_dict = {'coco': coco.CocoDataset, 'cityscapes': cityscapes.Dataset,
                            'cityscapesCoco': cityscapesCoco.CityscapesCocoDataset,
                            'kitti': kitti.KittiDataset, 'sbd': sbd.SbdDataset,
                            'pcb': pcb.PcbDataset, 'whu': whu.WhuDataset}
            dataset = dataset_dict[info['name']]
        dataset = dataset(info['anno_dir'], info['image_dir'], info['split'] if split is None else split, cfg, info['gt_image_dir'] if 'gt_image_dir' in info else None)
    return dataset


def make_data_sampler(dataset, shuffle, drop_last=False):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return make_ddp_data_sampler(dataset, shuffle, drop_last)
    else:
        return torch.utils.data.RandomSampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)

def make_ddp_data_sampler(dataset, shuffle, drop_last=False):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=torch.distributed.get_world_size(),
                                                              rank=torch.distributed.get_rank(),
                                                              shuffle=shuffle,
                                                              drop_last=drop_last)
    return sampler

def make_batch_data_sampler(sampler, batch_size, drop_last):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    return batch_sampler

def make_train_loader(cfg, split=None):
    batch_size = cfg.train.batch_size
    shuffle = True
    drop_last = False
    dataset_name = cfg.train.dataset

    if split in ('val', 'mini', 'test'):
        dataset = make_dataset(dataset_name, is_test=True, cfg=cfg, split=split)
    else:
        dataset = make_dataset(dataset_name, is_test=False, cfg=cfg, split=split)

    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    shuffle_flag = (not use_ddp)

    sampler = make_data_sampler(dataset, shuffle=shuffle_flag, drop_last=drop_last)
    num_workers = cfg.train.num_workers
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # batch_sampler=batch_sampler,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers = False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.commen.seed)  # DataLoader 전체 seed
    )
    return data_loader

def make_test_loader(cfg, is_distributed=True, split=None):
    batch_size = cfg.test.batch_size
    # shuffle = True if is_distributed else False
    shuffle = False
    drop_last = False
    dataset_name = cfg.test.dataset

    # dataset = make_dataset(dataset_name, is_test=True, cfg=cfg)
    if split in ('val','mini','test'):
        dataset = make_dataset(dataset_name, is_test=True, cfg=cfg, split=split)
    else:
        dataset = make_dataset(dataset_name, is_test=True, cfg=cfg)

    sampler = make_data_sampler(dataset, shuffle, drop_last=drop_last)
    # batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = 1
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # batch_sampler=batch_sampler,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.commen.seed),
        drop_last=drop_last
    )
    return data_loader


def make_data_loader(is_train=True, is_distributed=False, cfg=None, val_split=None):
    # print("[DEBUG] make_data_loader CALLED")
    # print("[DEBUG] make_data_loader CALLED", flush=True)
    if is_train:
        # print("[DEBUG] returning DataLoader (another condition)", flush=True)
        return make_train_loader(cfg), make_test_loader(cfg, split=val_split)
    else:
        # print("[DEBUG] returning DataLoader", flush=True)
        return make_test_loader(cfg, is_distributed)

def make_demo_loader(data_root=None, cfg=None):
    from .demo_dataset import Dataset
    batch_size = 1
    shuffle = False
    drop_last = False
    dataset = Dataset(data_root, cfg)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = 1
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )
    return data_loader

def make_ddp_train_loader(cfg, split=None):
    batch_size = cfg.train.batch_size
    shuffle = True
    drop_last = False
    dataset_name = cfg.train.dataset

    # dataset = make_dataset(dataset_name, is_test=False, cfg=cfg)
    if split in ('val', 'mini', 'test'):
        dataset = make_dataset(dataset_name, is_test=True, cfg=cfg, split=split)
    else:
        dataset = make_dataset(dataset_name, is_test=False, cfg=cfg, split=split)

    sampler = make_ddp_data_sampler(dataset, shuffle)
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        collate_fn=collator,
        pin_memory=False,
        drop_last=drop_last
    )
    return data_loader

def make_ddp_data_loader(is_train=True, is_distributed=False, cfg=None, val_split=None):
    if is_train:
        return make_ddp_train_loader(cfg), make_test_loader(cfg, split='val' if val_split is None else val_split)
    else:
        return make_test_loader(cfg, is_distributed)

