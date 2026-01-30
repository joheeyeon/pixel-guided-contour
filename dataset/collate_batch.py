from torch.utils.data.dataloader import default_collate
import torch
import numpy as np


def collate_batch(batch):
    data_input = {}
    # print(type(batch))  # 아마 list가 아니라 dict 나올 가능성 큼
    # print(batch.keys())  # dict면 이거 찍어봐
    # if isinstance(batch, dict): print(batch.keys())
    # if isinstance(batch, list): print(type(batch[0]))

    if isinstance(batch, list):
        if 'pys' in batch[0]:
            pys = {'pys': default_collate([b['pys'] for b in batch])}
            data_input.update(pys)
            ct_img_idx = {'ct_img_idx': default_collate([b['ct_img_idx'] for b in batch])}
            data_input.update(ct_img_idx)
        else:
            inp = {'inp': default_collate([b['inp'] for b in batch])}
            data_input.update(inp)
        if 'pixel_gt' in batch[0]:
            data_input.update({'pixel_gt': default_collate([b['pixel_gt'] for b in batch])})
        meta = default_collate([b['meta'] for b in batch])
        data_input.update({'meta': meta})

        # collate pyrnn
        if 'poly_label_array' in batch[0]:
            poly_label_array = default_collate([b['poly_label_array'] for b in batch])
            poly_index_target = default_collate([b['poly_index_target'] for b in batch])
            pyrnn = {'poly_label_array': poly_label_array, 'poly_index_target': poly_index_target}
            data_input.update(pyrnn)

        #collate detection
        if 'ct_hm' in batch[0]:
            ct_hm = default_collate([b['ct_hm'] for b in batch])
            max_len = torch.max(meta['ct_num'])
            batch_size = len(batch)
            wh = torch.zeros([batch_size, max_len, 2], dtype=torch.float)
            ct_cls = torch.zeros([batch_size, max_len], dtype=torch.int64)
            ct_ind = torch.zeros([batch_size, max_len], dtype=torch.int64)
            ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool)
            ct_img_idx = torch.zeros([batch_size, max_len], dtype=torch.int64)
            for i in range(batch_size):
                ct_01[i, :meta['ct_num'][i]] = 1
                ct_img_idx[i, :meta['ct_num'][i]] = i

            if max_len != 0:
                # wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
                # # reg[ct_01] = torch.Tensor(sum([b['reg'] for b in batch], []))
                # ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
                # ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))
                ct_ind_data = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))
                pad_ct_ind = torch.zeros((batch_size, max_len), dtype=torch.int64)
                pad_ct_ind[ct_01] = ct_ind_data
                ct_ind = pad_ct_ind

                ct_cls_data = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
                pad_ct_cls = torch.zeros((batch_size, max_len), dtype=torch.int64)
                pad_ct_cls[ct_01] = ct_cls_data
                ct_cls = pad_ct_cls

                wh_data = torch.Tensor(sum([b['wh'] for b in batch], []))
                pad_wh = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
                pad_wh[ct_01] = wh_data
                wh = pad_wh

            detection = {'ct_hm': ct_hm, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01, 'ct_img_idx': ct_img_idx, 'wh': wh}
            data_input.update(detection)

            # ✅ Pixel 태스크: contour 관련 collate 건너뜀
            if 'num_points' in batch[0]:
                #collate sementation
                num_points_per_poly = batch[0]['num_points']
                num_points_per_init = batch[0]['num_points_init']
                num_points_per_coarse = batch[0]['num_points_coarse']
                img_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
                can_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
                keyPointsMask = torch.zeros([batch_size, max_len, num_points_per_poly], dtype=torch.float)
                img_gt_coarse_polys = torch.zeros([batch_size, max_len, num_points_per_coarse, 2], dtype=torch.float)
                can_gt_coarse_polys = torch.zeros([batch_size, max_len, num_points_per_coarse, 2], dtype=torch.float)
                if 'img_it_init_polys' in batch[0]:
                    img_gt_init_polys = torch.zeros([batch_size, max_len, 4, 2], dtype=torch.float)
                    can_gt_init_polys = torch.zeros([batch_size, max_len, 4, 2], dtype=torch.float)

                    img_it_init_polys = torch.zeros([batch_size, max_len, num_points_per_init, 2], dtype=torch.float)
                    can_it_init_polys = torch.zeros([batch_size, max_len, num_points_per_init, 2], dtype=torch.float)
                    img_it_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
                    can_it_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
                else:
                    img_gt_init_polys = torch.zeros([batch_size, max_len, num_points_per_init, 2], dtype=torch.float)
                    can_gt_init_polys = torch.zeros([batch_size, max_len, num_points_per_init, 2], dtype=torch.float)

                if max_len != 0:
                    if 'img_it_init_polys' in batch[0]:
                        img_it_init_polys[ct_01] = torch.Tensor(np.array(sum([b['img_it_init_polys'] for b in batch], [])))
                        can_it_init_polys[ct_01] = torch.Tensor(np.array(sum([b['can_it_init_polys'] for b in batch], [])))
                        img_it_polys[ct_01] = torch.Tensor(np.array(sum([b['img_it_polys'] for b in batch], [])))
                        can_it_polys[ct_01] = torch.Tensor(np.array(sum([b['can_it_polys'] for b in batch], [])))
                    img_gt_polys[ct_01] = torch.Tensor(np.array(sum([b['img_gt_polys'] for b in batch], [])))
                    can_gt_polys[ct_01] = torch.Tensor(np.array(sum([b['can_gt_polys'] for b in batch], [])))
                    img_gt_init_polys[ct_01] = torch.Tensor(np.array(sum([b['img_gt_init_polys'] for b in batch], [])))
                    can_gt_init_polys[ct_01] = torch.Tensor(np.array(sum([b['can_gt_init_polys'] for b in batch], [])))
                    img_gt_coarse_polys[ct_01] = torch.Tensor(np.array(sum([b['img_gt_coarse_polys'] for b in batch], [])))
                    can_gt_coarse_polys[ct_01] = torch.Tensor(np.array(sum([b['can_gt_coarse_polys'] for b in batch], [])))
                    if 'keypoints_mask' in batch[0]:
                        keyPointsMask[ct_01] = torch.Tensor(np.array(sum([b['keypoints_mask'] for b in batch], [])))

                data_input.update({'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys,
                                   'img_gt_init_polys': img_gt_init_polys, 'can_gt_init_polys': can_gt_init_polys,
                                   'img_gt_coarse_polys': img_gt_coarse_polys, 'can_gt_coarse_polys': can_gt_coarse_polys})
                if 'img_it_init_polys' in batch[0]:
                    data_input.update({'img_it_init_polys': img_it_init_polys, 'can_it_init_polys': can_it_init_polys, 'img_it_polys': img_it_polys, 'can_it_polys': can_it_polys})
                if 'keypoints_mask' in batch[0]:
                    data_input.update({'keypoints_mask': keyPointsMask})

    # ✅ dict인 경우 (이미 collate 완료 상태)
    elif isinstance(batch, dict):
        # ✅ dict 내부 list/ndarray를 tensor로 변환
        for k, v in batch.items():
            if isinstance(v, list):
                v = np.array(v)  # 리스트 안에 리스트가 있을 수 있으니 np.array 먼저
            if isinstance(v, np.ndarray):
                # 정수 배열은 long으로, 실수 배열은 float32로
                if np.issubdtype(v.dtype, np.integer):
                    batch[k] = torch.from_numpy(v).long()
                else:
                    batch[k] = torch.from_numpy(v).float()

        if isinstance(batch.get('meta'), dict):
            for mk, mv in batch['meta'].items():
                if isinstance(mv, np.ndarray):
                    batch['meta'][mk] = torch.from_numpy(mv)
                elif isinstance(mv, list) and len(mv) > 0 and isinstance(mv[0], (int, float)):
                    batch['meta'][mk] = torch.tensor(mv)

        data_input.update(batch)

        meta = batch['meta']
        ct_num = meta['ct_num']
        if isinstance(ct_num, list):
            ct_num = torch.tensor(ct_num)
        elif isinstance(ct_num, int):
            ct_num = torch.tensor([ct_num])
        batch_size = ct_num.shape[0] if ct_num.ndim > 0 else 1
        max_len = torch.max(ct_num) if ct_num.ndim > 0 else ct_num

        wh = torch.zeros([batch_size, max_len, 2], dtype=torch.float)
        ct_cls = torch.zeros([batch_size, max_len], dtype=torch.int64)
        ct_ind = torch.zeros([batch_size, max_len], dtype=torch.int64)
        ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool)
        ct_img_idx = torch.zeros([batch_size, max_len], dtype=torch.int64)

        for i in range(batch_size):
            ct_01[i, :ct_num[i]] = 1
            ct_img_idx[i, :ct_num[i]] = i

        if max_len != 0:
            wh_data = torch.as_tensor(batch['wh'], dtype=torch.float32)
            ct_cls_data = torch.as_tensor(batch['ct_cls'], dtype=torch.int64)
            ct_ind_data = torch.as_tensor(batch['ct_ind'], dtype=torch.int64)

            pad_ct_cls = torch.zeros((batch_size, max_len), dtype=torch.int64)
            pad_wh = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)

            pad_ct_cls[ct_01] = ct_cls_data
            pad_wh[ct_01] = wh_data.reshape(-1, 2)

            ct_cls = pad_ct_cls
            wh = pad_wh

            if ct_ind_data.ndim == 1:
                # N → (batch_size, max_len)로 패딩
                pad_ct_ind = torch.zeros((batch_size, max_len), dtype=torch.int64)
                pad_ct_ind[ct_01] = ct_ind_data
                ct_ind = pad_ct_ind
            else:
                ct_ind = ct_ind_data

        data_input.update({
            'ct_cls': ct_cls,
            'ct_ind': ct_ind,
            'ct_01': ct_01,
            'ct_img_idx': ct_img_idx,
            'wh': wh,
        })
        if 'ct_hm' in batch:
            ct_hm = batch['ct_hm']
            if ct_hm.dim() == 3:  # (N, H, W)
                ct_hm = ct_hm.unsqueeze(1)  # (N, 1, H, W)
            data_input['ct_hm'] = ct_hm

        num_points_per_poly = int(batch['num_points']) if torch.is_tensor(batch['num_points']) else batch['num_points']
        num_points_per_init = int(batch['num_points_init']) if torch.is_tensor(batch['num_points_init']) else batch[
            'num_points_init']
        num_points_per_coarse = int(batch['num_points_coarse']) if torch.is_tensor(batch['num_points_coarse']) else \
        batch['num_points_coarse']

        img_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
        can_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
        img_gt_coarse_polys = torch.zeros([batch_size, max_len, num_points_per_coarse, 2], dtype=torch.float)
        can_gt_coarse_polys = torch.zeros([batch_size, max_len, num_points_per_coarse, 2], dtype=torch.float)

        if max_len != 0:
            img_data = torch.as_tensor(batch['img_gt_polys'], dtype=torch.float32)
            can_data = torch.as_tensor(batch['can_gt_polys'], dtype=torch.float32)
            img_coarse_data = torch.as_tensor(batch['img_gt_coarse_polys'], dtype=torch.float32)
            can_coarse_data = torch.as_tensor(batch['can_gt_coarse_polys'], dtype=torch.float32)

            flat_idx = 0
            for b in range(batch_size):
                n = int(ct_num[b])
                if n > 0:
                    img_gt_polys[b, :n] = img_data[flat_idx:flat_idx + n]
                    can_gt_polys[b, :n] = can_data[flat_idx:flat_idx + n]
                    img_gt_coarse_polys[b, :n] = img_coarse_data[flat_idx:flat_idx + n]
                    can_gt_coarse_polys[b, :n] = can_coarse_data[flat_idx:flat_idx + n]
                    flat_idx += n

            # img_gt_coarse_polys[ct_01] = batch['img_gt_coarse_polys'][ct_01]
            # can_gt_coarse_polys[ct_01] = batch['can_gt_coarse_polys'][ct_01]

        data_input.update({
            'img_gt_polys': img_gt_polys,
            'can_gt_polys': can_gt_polys,
            'img_gt_coarse_polys': img_gt_coarse_polys,
            'can_gt_coarse_polys': can_gt_coarse_polys,
        })

    return data_input
