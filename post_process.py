import torch
import numpy as np
from vertex_reordering import vertex_reordering_pipeline, has_self_intersection

def compute_num(offset_0, offset_1, thre=3):
    offset_0_front = torch.roll(offset_0, shifts=1, dims=1)
    offset_1_front = torch.roll(offset_1, shifts=1, dims=1)
    offset_0_front_2 = torch.roll(offset_0, shifts=2, dims=1)
    offset_1_front_2 = torch.roll(offset_1, shifts=2, dims=1)
    cos_0 = torch.sum(offset_0 * offset_0_front, dim=2)
    cos_1 = torch.sum(offset_1 * offset_1_front, dim=2)
    cos_0_2 = torch.sum(offset_0 * offset_0_front_2, dim=2)
    cos_1_2 = torch.sum(offset_1 * offset_1_front_2, dim=2)
    cos_0 = ((cos_0 < -0.1) & (cos_0_2 > 0.1)).to(torch.int)
    cos_1 = ((cos_1 < -0.1) & (cos_1_2 > 0.1)).to(torch.int)
    nums = (torch.sum(cos_1, dim=1) - torch.sum(cos_0, dim=1) >= thre).to(torch.int)
    nums = nums.unsqueeze(1).unsqueeze(2).expand(offset_0.size(0), offset_0.size(1), offset_0.size(2))
    return nums

def post_process(output):
    end_py = output['py'][-1].detach()
    gcn_py = output['py'][-2].detach()
    
    if len(end_py) == 0:
        return 0
    
    offset_1 = end_py - torch.roll(end_py, shifts=1, dims=1)
    offset_0 = gcn_py - torch.roll(gcn_py, shifts=1, dims=1)
    nokeep = compute_num(offset_0, offset_1)
    end_poly = end_py * (1 - nokeep) + gcn_py * nokeep
    output['py'].append(end_poly)
    return 0


def vertex_reordering_post_process(contours, method='auto', apply_threshold=True):
    """
    Contour들에 대해 vertex re-ordering 후처리 적용
    
    Args:
        contours: List of (V, 2) torch.Tensor or numpy arrays
        method: reordering method ('auto', 'local_triangle', 'angle_based', etc.)
        apply_threshold: True이면 self-intersection이 있는 경우에만 적용
    
    Returns:
        reordered_contours: List of reordered contours
    """
    if not isinstance(contours, (list, tuple)):
        contours = [contours]

    reordered_contours = []
    stats = {'total': len(contours), 'processed': 0, 'improved': 0}

    for i, contour in enumerate(contours):
        try:
            # torch tensor를 numpy로 변환
            if isinstance(contour, torch.Tensor):
                contour_np = contour.detach().cpu().numpy()
                return_torch = True
                original_device = contour.device
                original_dtype = contour.dtype
            else:
                contour_np = np.array(contour)
                return_torch = False
                original_device = None
                original_dtype = None
            
            
            # 차원 처리 - 안전한 방식으로 변경
            while len(contour_np.shape) > 2:
                if contour_np.shape[0] == 1:
                    contour_np = contour_np[0]  # (1, V, 2) -> (V, 2)
                elif contour_np.shape[-1] == 2:
                    # 마지막 차원이 2 (좌표)인 경우, 앞 차원들을 flatten
                    contour_np = contour_np.reshape(-1, 2)
                    break
                else:
                    # 첫 번째 요소만 선택
                    contour_np = contour_np[0]
            
            
            # 점이 너무 적으면 건너뛰기
            if len(contour_np) < 3:
                if return_torch:
                    reordered_contours.append(contour)
                else:
                    reordered_contours.append(contour_np)
                continue
            
            # self-intersection 체크 및 처리 여부 결정
            has_intersection_before = has_self_intersection(contour_np)
            
            if apply_threshold and not has_intersection_before:
                # self-intersection이 없으면 원본 반환
                if return_torch:
                    reordered_contours.append(contour)
                else:
                    reordered_contours.append(contour_np)
                continue
            
            
            # vertex re-ordering 적용
            reordered = vertex_reordering_pipeline(contour_np, method=method)
            stats['processed'] += 1
            
            
            # reordering 결과가 너무 적은 점을 가지면 원본 반환
            if len(reordered) < 3:
                if return_torch:
                    reordered_contours.append(contour)
                else:
                    reordered_contours.append(contour_np)
                continue
            
            # 결과 검증
            has_intersection_after = has_self_intersection(reordered)
            
            if has_intersection_before and not has_intersection_after:
                stats['improved'] += 1
            
            # 원래 format으로 변환해서 반환
            if return_torch:
                reordered_tensor = torch.from_numpy(reordered).to(original_device).type(original_dtype)
                reordered_contours.append(reordered_tensor)
            else:
                reordered_contours.append(reordered)
                
        except Exception as e:
            print(f"[Warning] Vertex reordering failed for contour {i}: {str(e)}")
            # 실패시 원본 반환
            reordered_contours.append(contour)
    
    
    return reordered_contours


def apply_vertex_reordering_to_output(output, method='auto', stage='final'):
    """
    네트워크 output에 vertex re-ordering 적용
    
    Args:
        output: 네트워크 output dictionary
        method: reordering method
        stage: 적용할 stage ('final', 'all', or specific stage name)
    
    Returns:
        output: 수정된 output dictionary
    """
    if 'py' not in output or len(output['py']) == 0:
        return output
    
    if stage == 'final':
        # 마지막 단계에만 적용
        final_contours = output['py'][-1]
        if len(final_contours) > 0:
            # ✅ tensor를 리스트로 변환해서 전달
            if isinstance(final_contours, torch.Tensor):
                contour_list = [final_contours[i] for i in range(final_contours.shape[0])]
            else:
                contour_list = final_contours
            reordered = vertex_reordering_post_process(contour_list, method=method)
            
            
            # 원래 batch 형태 복원
            
            if isinstance(reordered[0], torch.Tensor):
                # 각 contour의 vertex 수가 달라질 수 있으므로 원래 형태에 맞춰 처리
                original_shape = final_contours.shape  # (N, V, 2)
                
                if len(original_shape) == 3:
                    # 원래 형태가 (N, V, 2)인 경우
                    if len(reordered) == 1:
                        # 단일 contour: vertex 수 맞춤
                        reordered_tensor = reordered[0]  # (V_new, 2)
                        if reordered_tensor.shape[0] < original_shape[1]:
                            # vertex 수가 줄어든 경우: padding으로 채움 (마지막 점으로)
                            last_point = reordered_tensor[-1:].repeat(original_shape[1] - reordered_tensor.shape[0], 1)
                            reordered_tensor = torch.cat([reordered_tensor, last_point], dim=0)
                        elif reordered_tensor.shape[0] > original_shape[1]:
                            # vertex 수가 늘어난 경우: 원래 수로 자름
                            reordered_tensor = reordered_tensor[:original_shape[1]]
                        output['py'][-1] = reordered_tensor.unsqueeze(0)  # (1, V, 2)
                    else:
                        # 다중 contour: 각각 처리
                        processed_contours = []
                        for j, r_contour in enumerate(reordered):
                            if j < original_shape[0]:  # 원래 contour 수 초과하지 않도록
                                if r_contour.shape[0] < original_shape[1]:
                                    # vertex 수 맞춤
                                    last_point = r_contour[-1:].repeat(original_shape[1] - r_contour.shape[0], 1)
                                    r_contour = torch.cat([r_contour, last_point], dim=0)
                                elif r_contour.shape[0] > original_shape[1]:
                                    r_contour = r_contour[:original_shape[1]]
                                processed_contours.append(r_contour.unsqueeze(0))
                        
                        if processed_contours:
                            output['py'][-1] = torch.cat(processed_contours, dim=0)  # (N, V, 2)
                        else:
                            # 빈 경우 원래 값 유지
                            pass
                else:
                    # 2D tensor인 경우: 각 polygon을 separate하게 유지
                    if len(reordered) == 1:
                        output['py'][-1] = reordered[0]  # 단일 polygon
                    else:
                        # 다중 polygon: stack으로 batch 차원 유지 (cat이 아닌 stack 사용)
                        try:
                            output['py'][-1] = torch.stack(reordered, dim=0)  # (N, V, 2)
                        except RuntimeError as e:
                            # vertex 수가 다를 경우 최대 vertex 수로 padding
                            max_vertices = max(r.shape[0] for r in reordered)
                            padded_contours = []
                            for i, r_contour in enumerate(reordered):
                                if r_contour.shape[0] < max_vertices:
                                    # 부족한 vertex는 마지막 점으로 padding
                                    last_point = r_contour[-1:].repeat(max_vertices - r_contour.shape[0], 1)
                                    r_contour = torch.cat([r_contour, last_point], dim=0)
                                padded_contours.append(r_contour)
                            output['py'][-1] = torch.stack(padded_contours, dim=0)  # (N, V, 2)
            else:
                output['py'][-1] = reordered
            
    elif stage == 'all':
        # 모든 단계에 적용
        for i, stage_contours in enumerate(output['py']):
            if len(stage_contours) > 0:
                # ✅ tensor를 리스트로 변환해서 전달
                if isinstance(stage_contours, torch.Tensor):
                    contour_list = [stage_contours[j] for j in range(stage_contours.shape[0])]
                else:
                    contour_list = stage_contours
                reordered = vertex_reordering_post_process(contour_list, method=method)
                # ✅ 원래 batch 형태 복원 (final과 동일한 로직)
                if isinstance(reordered[0], torch.Tensor):
                    # 각 contour의 vertex 수가 달라질 수 있으므로 원래 형태에 맞춰 처리
                    original_shape = stage_contours.shape  # (N, V, 2)
                    if len(original_shape) == 3:
                        # 원래 형태가 (N, V, 2)인 경우
                        if len(reordered) == 1:
                            # 단일 contour: vertex 수 맞춤
                            reordered_tensor = reordered[0]  # (V_new, 2)
                            if reordered_tensor.shape[0] < original_shape[1]:
                                # vertex 수가 줄어든 경우: padding으로 채움 (마지막 점으로)
                                last_point = reordered_tensor[-1:].repeat(original_shape[1] - reordered_tensor.shape[0], 1)
                                reordered_tensor = torch.cat([reordered_tensor, last_point], dim=0)
                            elif reordered_tensor.shape[0] > original_shape[1]:
                                # vertex 수가 늘어난 경우: 원래 수로 자름
                                reordered_tensor = reordered_tensor[:original_shape[1]]
                            output['py'][i] = reordered_tensor.unsqueeze(0)  # (1, V, 2)
                        else:
                            # 다중 contour: 각각 처리
                            processed_contours = []
                            for j, r_contour in enumerate(reordered):
                                if j < original_shape[0]:  # 원래 contour 수 초과하지 않도록
                                    if r_contour.shape[0] < original_shape[1]:
                                        # vertex 수 맞춤
                                        last_point = r_contour[-1:].repeat(original_shape[1] - r_contour.shape[0], 1)
                                        r_contour = torch.cat([r_contour, last_point], dim=0)
                                    elif r_contour.shape[0] > original_shape[1]:
                                        r_contour = r_contour[:original_shape[1]]
                                    processed_contours.append(r_contour.unsqueeze(0))
                            
                            if processed_contours:
                                output['py'][i] = torch.cat(processed_contours, dim=0)  # (N, V, 2)
                            else:
                                # 빈 경우 원래 값 유지
                                pass
                    else:
                        # 2D tensor인 경우: 각 polygon을 separate하게 유지
                        if len(reordered) == 1:
                            output['py'][i] = reordered[0]  # 단일 polygon
                        else:
                            # 다중 polygon: stack으로 batch 차원 유지 (cat이 아닌 stack 사용)
                            try:
                                output['py'][i] = torch.stack(reordered, dim=0)  # (N, V, 2)
                            except RuntimeError as e:
                                # vertex 수가 다를 경우 최대 vertex 수로 padding
                                max_vertices = max(r.shape[0] for r in reordered)
                                padded_contours = []
                                for r_contour in reordered:
                                    if r_contour.shape[0] < max_vertices:
                                        # 부족한 vertex는 마지막 점으로 padding
                                        last_point = r_contour[-1:].repeat(max_vertices - r_contour.shape[0], 1)
                                        r_contour = torch.cat([r_contour, last_point], dim=0)
                                    padded_contours.append(r_contour)
                                output['py'][i] = torch.stack(padded_contours, dim=0)  # (N, V, 2)
                else:
                    output['py'][i] = reordered
    
    return output

