================================================================================
        MÔ TẢ ĐỒ ÁN
  ĐIỀU KHIỂN ĐÈN GIAO THÔNG THÍCH NGHI BẰNG HỌC TĂNG CƯỜNG (RL)
  VÀ HỆ ĐA TÁC TỬ (MAS) — MÔ PHỎNG TRÊN SUMO
  Khu vực: Ngã tư Thủ Đức (N4TD), TP. Hồ Chí Minh
================================================================================


1. TÓM TẮT ĐỒ ÁN
--------------------------------------------------------------------------------
Đồ án xây dựng một hệ thống điều khiển đèn tín hiệu giao thông THÔNG MINH,
có khả năng TỰ HỌC và TỰ THÍCH NGHI theo lưu lượng xe thực tế, thay vì điều
khiển theo chu kỳ đèn cố định như hiện nay.

Hệ thống được mô phỏng trên phần mềm SUMO (Simulation of Urban MObility) qua
giao tiếp TraCI, áp dụng cho hai nút đèn liền kề tại khu vực Ngã tư Thủ Đức:
    - GL1: nút giao chính  (12 pha đèn, 17 cảm biến)
    - GL2: nút giao phụ    ( 8 pha đèn,  6 cảm biến)

Mục tiêu là so sánh ba phương pháp điều khiển để chứng minh hiệu quả của
học tăng cường, đặc biệt khi các nút đèn biết PHỐI HỢP với nhau:
    (1) Đèn cố định (Fixed-Time)        — baseline truyền thống
    (2) Học tăng cường độc lập (Independent RL) — mỗi nút tự học riêng
    (3) Hệ đa tác tử phối hợp (Cooperative MAS) — hai nút chia sẻ thông tin


2. BÀI TOÁN VÀ ĐỘNG LỰC
--------------------------------------------------------------------------------
Đèn giao thông truyền thống chạy theo chu kỳ cố định, không phản ứng được khi
lưu lượng thay đổi (giờ cao điểm, hướng đường mất cân đối...). Điều này gây
ùn tắc, tăng thời gian chờ và giảm thông lượng xe qua nút.

Đồ án đặt ra câu hỏi: Liệu một tác tử (agent) học tăng cường có thể tự học cách
chuyển pha đèn hợp lý dựa trên số xe đang chờ, để giảm ùn tắc tốt hơn đèn cố
định hay không? Và nếu hai nút đèn gần nhau biết "trao đổi" tình trạng giao
thông cho nhau thì có cải thiện thêm được không?


3. MÔI TRƯỜNG MÔ PHỎNG (SUMO)
--------------------------------------------------------------------------------
- Mạng đường (N4TD_FS.net.xml) được dựng mô phỏng khu vực Ngã tư Thủ Đức thực tế.
- Hệ thống cảm biến: dùng detector loại E2 (lanearea) đặt trên các làn vào nút
  để đếm số xe đang dừng chờ (halting number) theo từng hướng.
- Đa dạng loại phương tiện sát thực tế Việt Nam: xe máy (motorbike), xe máy
  công nghệ (Grab), xe máy nặng, xe đạp, xe đạp điện, ô tô con, SUV, taxi,
  xe buýt, xe tải, xe van, xe ba gác, xe ưu tiên (emergency) — mỗi loại có
  kích thước, tốc độ, gia tốc riêng.

5 KỊCH BẢN GIAO THÔNG được thiết kế để kiểm thử trong nhiều điều kiện:
    - low_demand     : lưu lượng thấp   (~750 xe/giờ)
    - medium_demand  : lưu lượng trung bình (~1.750 xe/giờ)
    - peak_hour      : giờ cao điểm     (~3.000 xe/giờ)
    - asymmetric     : lưu lượng mất cân đối (hướng Tây→Đông gấp đôi chiều ngược lại)
    - mixed_vehicle  : đa dạng phương tiện (nhiều xe buýt, xe tải, taxi...)


4. MÔ HÌNH HỌC TĂNG CƯỜNG
--------------------------------------------------------------------------------
Bài toán được mô hình hóa theo dạng MDP (Markov Decision Process):

- TRẠNG THÁI (State): vector gồm số xe đang chờ trên các detector của nút,
  cộng thêm pha đèn hiện tại (đã chuẩn hóa).
      GL1: 18 chiều (17 detector + 1 pha)
      GL2:  7 chiều ( 6 detector + 1 pha)

- HÀNH ĐỘNG (Action): 2 lựa chọn
      0 = giữ nguyên pha hiện tại
      1 = chuyển sang pha xanh kế tiếp
  (Có ràng buộc thời gian xanh tối thiểu MIN_GREEN để đèn không nhấp nháy liên tục.)

- PHẦN THƯỞNG (Reward): r = − (tổng số xe đang chờ)
  Càng ít xe phải dừng chờ thì phần thưởng càng cao. Lựa chọn dùng số xe chờ
  (queue) thay vì thời gian chờ vì nó ổn định, ít nhiễu và dễ tối ưu hơn.


5. CÁC THUẬT TOÁN / TÁC TỬ ĐÃ TRIỂN KHAI
--------------------------------------------------------------------------------
(a) Q-Learning (bảng Q-table)
    - Rời rạc hóa trạng thái (số xe chờ chia thành các mức) rồi học bằng
      phương trình Bellman.
    - Chọn hành động theo chiến lược ε-greedy (thăm dò + khai thác).

(b) DQN — Deep Q-Network (mạng nơ-ron sâu)
    - Mạng nơ-ron xấp xỉ hàm Q, có Online Network + Target Network để ổn định.
    - Experience Replay (bộ nhớ kinh nghiệm) để học hiệu quả hơn.
    - Cải tiến kỹ thuật: thay BatchNormalization bằng LayerNormalization cho
      phù hợp với học online (batch nhỏ, dữ liệu không ổn định).

(c) Cooperative MAS — Hệ đa tác tử phối hợp (điểm nhấn của đồ án)
    - Mỗi nút đèn là một tác tử DQN riêng.
    - MessageBus (kênh trao đổi tin): nút phụ GL2 "phát" thông tin trạng thái
      (số xe chờ, thời gian chờ, pha đèn, mật độ) lên kênh chung; nút chính
      GL1 "nhận" và ghép vào trạng thái của mình (mở rộng 18 → 25 chiều) để
      ra quyết định có tính phối hợp.
    - RewardShaper (định hình phần thưởng): kết hợp phần thưởng cục bộ của từng
      nút với phần thưởng toàn mạng theo công thức
          r = −(local_queue) − 0.3 × (global_queue) − 0.1 × (waiting_penalty)
      giúp mỗi nút vừa tối ưu cho mình, vừa có động lực giảm tắc nghẽn cho cả mạng.
    - Áp dụng kỹ thuật nâng cao: Double DQN, soft-update target network,
      gradient clipping.

(d) Fixed-Time — Đèn cố định (baseline để so sánh)
    - Chạy theo chu kỳ pha cố định (GL1 ~90s, GL2 ~60s), không học, không thích nghi.


6. CẤU TRÚC MÃ NGUỒN
--------------------------------------------------------------------------------
    config.py                  : Cấu hình trung tâm (ID đèn, detector, siêu tham số)
    environment/
        sumo_env.py            : Môi trường SUMO, kết nối TraCI, đọc state/áp action
        reward.py              : Hàm tính phần thưởng
    agents/
        base_agent.py          : Lớp cơ sở chung cho mọi tác tử
        q_learning_agent.py    : Tác tử Q-Learning
        dqn_agent.py           : Tác tử DQN
        cooperative_agent.py   : Tác tử DQN có phối hợp MAS
    communication/
        message_bus.py         : Kênh trao đổi tin giữa các nút đèn
        reward_shaper.py       : Định hình phần thưởng cấp toàn mạng
    baselines/
        fixed_time.py          : Bộ điều khiển đèn cố định
    training/
        train_q.py             : Huấn luyện Q-Learning
        train_dqn.py           : Huấn luyện DQN
        train_mas.py           : Huấn luyện/so sánh 3 chế độ (fixed/independent/mas)
        evaluate.py            : Đánh giá mô hình
    experiments/
        run_experiments.py     : Tự động chạy toàn bộ thực nghiệm (3 mode × 5 kịch bản)
    evaluation/
        compare_results.py     : Tổng hợp kết quả, vẽ bảng & biểu đồ so sánh
    run_sim.py                 : Chạy lại (replay) chính sách đã học để quan sát/quay video
    run_all.py                 : Pipeline tổng (kiểm tra → train → benchmark)
    network/ , scenarios/ , cfg/ : File mạng đường, kịch bản, cấu hình SUMO
    models/                    : Lưu Q-table (.pkl) và trọng số DQN (.keras) đã train
    results/ , output/         : Biểu đồ, log và dữ liệu kết quả mô phỏng


7. QUY TRÌNH THỰC NGHIỆM
--------------------------------------------------------------------------------
1) Kiểm tra kết nối SUMO ↔ TraCI.
2) Huấn luyện các tác tử (Q-Learning / DQN / MAS) trên từng kịch bản.
   Mỗi cấu hình chạy nhiều lần (runs) với seed khác nhau để kết quả khách quan.
3) Lưu mô hình đã học vào thư mục models/.
4) Đánh giá ở chế độ tham lam (epsilon=0, chỉ khai thác, không học thêm).
5) So sánh ba phương pháp theo các chỉ số:
       - Thời gian chờ trung bình (Avg Waiting Time)
       - Chiều dài hàng chờ trung bình (Avg Queue Length)
       - Tốc độ trung bình (Avg Speed)
       - Thông lượng / số xe qua nút (Throughput)
       - Tổng phần thưởng (Total Reward)
6) Xuất bảng số liệu, biểu đồ học tập (learning curve) và biểu đồ benchmark.


8. KẾT QUẢ CHÍNH
--------------------------------------------------------------------------------
- Đã huấn luyện thành công và lưu được mô hình cho cả 5 kịch bản ở cả ba
  phương pháp (Q-Learning, DQN, và DQN phối hợp MAS).
- Biểu đồ benchmark (results/plots/benchmark_comparison.png) cho thấy các
  phương pháp học tăng cường giảm rõ rệt chiều dài hàng chờ trung bình so với
  đèn cố định, đặc biệt ở các kịch bản lưu lượng cao.
- Chế độ phối hợp MAS (hai nút chia sẻ thông tin) cho kết quả tốt hơn so với
  để mỗi nút học độc lập, chứng minh giá trị của việc phối hợp giữa các nút đèn
  liền kề.

  (Lưu ý: các con số cụ thể được lưu trong file kết quả JSON/CSV và các biểu đồ
   trong thư mục results/ và output/; có thể trình bày chi tiết trong báo cáo.)


9. ĐÓNG GÓP & ĐIỂM NỔI BẬT
--------------------------------------------------------------------------------
- Mô phỏng sát thực tế giao thông Việt Nam: hỗn hợp xe máy chiếm tỉ lệ lớn,
  nhiều loại phương tiện, áp dụng cho một nút giao thực tế (Ngã tư Thủ Đức).
- Triển khai đầy đủ từ thuật toán cơ bản (Q-Learning) đến nâng cao (DQN,
  Double DQN, MAS), có baseline đèn cố định để so sánh công bằng.
- Thiết kế cơ chế phối hợp đa tác tử (MessageBus + RewardShaper) — đây là
  điểm mới và là trọng tâm khoa học của đồ án.
- Mã nguồn có cấu trúc rõ ràng, tách module, dễ mở rộng và tái lập thí nghiệm.


10. CÔNG NGHỆ SỬ DỤNG
--------------------------------------------------------------------------------
- Ngôn ngữ: Python
- Mô phỏng giao thông: SUMO + TraCI
- Học sâu: TensorFlow / Keras
- Xử lý số & trực quan hóa: NumPy, Matplotlib

================================================================================