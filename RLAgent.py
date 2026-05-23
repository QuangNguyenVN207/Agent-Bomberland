import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. Định nghĩa Kiến trúc Não bộ (Neural Network) của AI
# =====================================================================
class DQNNetwork(nn.Module):
    """
    Mạng Neural Network nhận đầu vào là trạng thái bản đồ (Flatten Grid)
    và dự đoán "Giá trị tương lai" (Q-value) của 6 hành động có thể thực hiện.
    """
    def __init__(self, input_size, action_size):
        super(DQNNetwork, self).__init__()
        # Tạo các lớp ẩn (Hidden Layers) xử lý thông tin
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size) # Đầu ra: 6 hành động (0 đến 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# =====================================================================
# 2. Lớp RLAgent quản lý quá trình Chọn hành động và Học tập
# =====================================================================
class RLAgent:
    team_id = "RLBomberAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        
        # Cấu hình kích thước môi trường (Giả định bản đồ game cố định là 11x11)
        self.grid_width = 11
        self.grid_height = 11
        self.input_size = self.grid_width * self.grid_height # 121 ô số
        self.action_size = 6 # 0:Đứng yên, 1:Trái, 2:Phải, 3:Lên, 4:Xuống, 5:Đặt bom
        
        # Khởi tạo Mạng Neural chính và Mạng Đích (Target Network để ổn định việc học)
        self.policy_net = DQNNetwork(self.input_size, self.action_size)
        self.target_net = DQNNetwork(self.input_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Bộ tối ưu hóa (Optimizer) dùng để cập nhật trọng số não bộ khi học
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Tham số Khám phá (Epsilon-Greedy): Giúp AI cân bằng giữa thử nghiệm cái mới và dùng cái đã biết
        self.epsilon = 1.0       # Ban đầu chọn ngẫu nhiên 100% để khám phá thế giới
        self.epsilon_decay = 0.995 # Giảm dần tính ngẫu nhiên sau mỗi lượt (khôn dần lên)
        self.epsilon_min = 0.05   # Giữ lại ít nhất 5% ngẫu nhiên để tránh bị rập khuôn

        # Lưu lại trạng thái và hành động trước đó để phục vụ tính toán phần thưởng (Reward)
        self.last_state = None
        self.last_action = None

    def _preprocess(self, obs):
        """ Chuyển đổi dữ liệu thô của môi trường thành vector số phẳng (0.0 - 1.0) để nạp vào AI """
        grid = obs["map"]
        # Chuẩn hóa ma trận (Ví dụ chia cho giá trị lớn nhất của ô để đưa về khoảng 0 đến 1)
        normalized_grid = grid.astype(np.float32) / 4.0 
        return normalized_grid.flatten() # Biến ma trận 2D thành mảng 1D (121 phần tử)

    def act(self, obs):
        """ Hàm giao tiếp chính: Nhận obs từ game và trả về hành động của AI """
        state = self._preprocess(obs)
        
        # Chuyển đổi trạng thái sang dạng Tensor của PyTorch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Chiến thuật Epsilon-Greedy (Thử và Sai)
        if random.random() < self.epsilon:
            # Giai đoạn đầu / May rủi: Chọn đại một hành động ngẫu nhiên để học
            action = random.randint(0, self.action_size - 1)
        else:
            # Giai đoạn thông minh: Đưa trạng thái vào Não bộ, chọn hành động có điểm dự đoán cao nhất
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = torch.argmax(q_values).item()

        # Lưu lại để dùng cho hàm học tập (train) ở cuối lượt
        self.last_state = state
        self.last_action = action
        
        return action

    # =====================================================================
    # 3. Hàm Học Tập (Học tăng cường dựa trên Phần thưởng)
    # =====================================================================
    def learn(self, current_obs, reward, done):
        """ 
        Hàm này sẽ được gọi sau khi hành động đã thực hiện xong.
        Nó giúp AI nhìn lại: "Trạng thái cũ -> Làm việc đó -> Nhận được bao nhiêu Điểm thưởng?"
        """
        if self.last_state is None or self.last_action is None:
            return

        # 1. Tiền xử lý trạng thái hiện tại (sau khi đã thực hiện hành động)
        next_state = self._preprocess(current_obs)

        # 2. Chuyển đổi toàn bộ dữ liệu sang PyTorch Tensor
        state_t = torch.FloatTensor(self.last_state)
        next_state_t = torch.FloatTensor(next_state)
        reward_t = torch.FloatTensor([reward])
        action_t = torch.LongTensor([self.last_action])

        # 3. Tính toán giá trị Q hiện tại từ mạng não bộ
        current_q = self.policy_net(state_t)[action_t]

        # 4. Tính toán giá trị Q kỳ vọng ở tương lai (Sử dụng công thức Bellman Equation)
        with torch.no_grad():
            if done:
                target_q = reward_t # Nếu chết game / hết giờ, không còn tương lai
            else:
                # Tương lai = Phần thưởng hiện tại + 0.99 * (Giá trị lớn nhất của bước đi kế tiếp)
                target_q = reward_t + 0.99 * torch.max(self.target_net(next_state_t))

        # 5. Tính toán độ lỗi (Loss) giữa dự đoán và thực tế
        loss = nn.MSELoss()(current_q, target_q)

        # 6. Lan truyền ngược và cập nhật lại trọng số "não bộ"
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Giảm dần độ ngẫu nhiên (Epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Cập nhật mạng Đích (Target Net) định kỳ (ví dụ sau mỗi lượt hoặc mỗi trận)
        # Để đơn giản, ở đây cập nhật nhẹ (Soft Update) sau mỗi bước đi
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(0.005 * policy_param.data + (1.0 - 0.005) * target_param.data)