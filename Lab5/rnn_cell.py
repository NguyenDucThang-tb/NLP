"""Simple RNN cell tự cài bằng NumPy cho LAB 05."""

import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=2026):
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size và hidden_size phải là số nguyên dương")

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Dùng generator riêng với seed cố định để khởi tạo có thể lặp lại.
        rng = np.random.default_rng(seed)
        self.W_xh = rng.uniform(-0.1, 0.1, size=(hidden_size, input_size))
        self.W_hh = rng.uniform(-0.1, 0.1, size=(hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size, dtype=np.float64)

    def forward(self, x_t, h_prev):
        x_t = np.asarray(x_t, dtype=np.float64)
        h_prev = np.asarray(h_prev, dtype=np.float64)

        if x_t.shape != (self.input_size,):
            raise ValueError(
                f"x_t phải có shape ({self.input_size},), nhận được {x_t.shape}"
            )
        if h_prev.shape != (self.hidden_size,):
            raise ValueError(
                f"h_prev phải có shape ({self.hidden_size},), nhận được {h_prev.shape}"
            )

        # Tính ảnh hưởng của input hiện tại.
        input_contribution = self.W_xh @ x_t
        # Tính ảnh hưởng của hidden state trước đó.
        recurrent_contribution = self.W_hh @ h_prev
        # Cộng hai nguồn thông tin và bias.
        pre_activation = input_contribution + recurrent_contribution + self.b_h
        # Dùng tanh để tính hidden state mới.
        h_t = np.tanh(pre_activation)
        return h_t
