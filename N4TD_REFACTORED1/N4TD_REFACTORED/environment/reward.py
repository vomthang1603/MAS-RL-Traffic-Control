# =============================================================================
# reward.py — Hàm Reward cho bài toán điều khiển đèn N4TD
#
# Tách biệt khỏi SumoEnv để dễ thử nghiệm và thay thế hàm reward
# mà không cần chỉnh sửa phần kết nối TraCI.
#
# Reward hiện tại:
#   r = −(tổng số xe đang chờ trên tất cả lane vào nút)
#
# Lý do chọn queue count (không phải waiting time):
#   - Đơn giản, ổn định, dễ debug
#   - Tương đương waiting time về mặt thứ tự tối ưu
#   - Ít nhiễu hơn tốc độ trung bình khi làn trống
# =============================================================================

import traci


def compute_reward(det_ids: list, tl_id: str) -> tuple:
    """
    Tính reward và thông tin cho 1 nút đèn giao thông.

    Params
    ------
    det_ids : list[str]  — danh sách ID detector e2 của nút
    tl_id   : str        — ID đèn tín hiệu (dùng cho info)

    Returns
    -------
    reward : float  — âm tổng queue (cao hơn = tốt hơn)
    info   : dict   — {'tl', 'total_queue', 'total_wait'}
    """
    total_wait  = 0.0
    total_queue = 0

    for did in det_ids:
        total_wait  += traci.lanearea.getLastStepMeanSpeed(did)   # m/s
        total_queue += traci.lanearea.getLastStepHaltingNumber(did)

    reward = -float(total_queue)

    info = {
        'tl'          : tl_id,
        'total_queue' : total_queue,
        'total_wait'  : total_wait,
    }
    return reward, info
