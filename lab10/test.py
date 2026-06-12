import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 实现 Sinusoidal Position Encoding 正弦位置编码 =====================
def sinusoidal_pos_encoding(seq_len: int, dim: int, device: torch.device = torch.device("cpu")):
    """
    标准 Transformer 正弦位置编码
    :param seq_len: 序列长度
    :param dim: 词嵌入维度(必须为偶数)
    :return: pos_enc: [seq_len, dim]
    """
    assert dim % 2 == 0, "维度必须是偶数"
    pos = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
    # 计算频率项
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim)
    )  # [dim/2]
    pos_enc = torch.zeros((seq_len, dim), device=device)
    pos_enc[:, 0::2] = torch.sin(pos * div_term)   # 偶数位 sin
    pos_enc[:, 1::2] = torch.cos(pos * div_term)   # 奇数位 cos
    return pos_enc

# ===================== 2. 实现二维向量旋转 =====================
def rotate_2d(vec: torch.Tensor, theta: float) -> torch.Tensor:
    """
    二维向量旋转
    :param vec: 输入二维向量 [2]
    :param theta: 旋转角度(弧度)
    :return: 旋转后向量 [2]
    """
    x, y = vec[0], vec[1]
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # 二维旋转矩阵: [cosθ, -sinθ; sinθ, cosθ]
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return torch.tensor([x_rot, y_rot])

# ===================== 3. 实现高维 RoPE (Rotary Position Embedding) =====================
def rope_embedding(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    高维旋转位置编码 RoPE
    :param x: 输入向量 [seq_len, dim] / [bs, seq_len, dim]
    :param pos: 位置索引 [seq_len]
    :return: 施加RoPE后的向量，维度同输入
    """
    dim = x.size(-1)
    assert dim % 2 == 0, "RoPE要求维度为偶数"

    # 构造旋转角度 θ = pos / 10000^(2i/dim)
    theta_base = 10000.0
    theta = pos.unsqueeze(-1) * torch.exp(
        torch.arange(0, dim, device=x.device, step=2) * (-math.log(theta_base) / dim)
    )  # [seq_len, dim/2]

    # 奇偶拆分
    x1 = x[..., 0::2]  # 偶数维度 [seq_len, dim/2]
    x2 = x[..., 1::2]  # 奇数维度 [seq_len, dim/2]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # RoPE 旋转公式
    x1_rot = x1 * cos_theta - x2 * sin_theta
    x2_rot = x1 * sin_theta + x2 * cos_theta

    # 拼接回原维度
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x1_rot
    x_rot[..., 1::2] = x2_rot
    return x_rot

# ===================== 4. 对比 E+pos 与 RoPE 输入方式 =====================
def compare_input_mode():
    print("=" * 50)
    print("4. E+pos 与 RoPE 输入方式对比")
    print("=" * 50)
    seq_len = 5
    dim = 8
    # 1. 词嵌入 E
    E = torch.randn(seq_len, dim)
    # 2. 正弦位置编码 pos_enc
    pos_enc = sinusoidal_pos_encoding(seq_len, dim)

    # -------- 方式1: E + pos (传统加法式位置编码) --------
    E_plus_pos = E + pos_enc
    print(f"词嵌入 E shape: {E.shape}")
    print(f"位置编码 pos_enc shape: {pos_enc.shape}")
    print(f"E + pos 结果 shape: {E_plus_pos.shape}")
    print("E+pos 逻辑: 词嵌入向量 逐元素 + 位置向量，位置信息直接叠加到内容向量")

    # -------- 方式2: RoPE (旋转式位置编码) --------
    pos_idx = torch.arange(seq_len)
    E_rope = rope_embedding(E, pos_idx)
    print(f"\nRoPE 处理后向量 shape: {E_rope.shape}")
    print("RoPE 逻辑: 不对向量做加法，而是**按位置对向量做旋转变换**，位置信息融入旋转角度")
    print("=" * 50 + "\n")

# ===================== 5. 数值实验：验证 RoPE 相对位置性质 =====================
def verify_relative_position():
    print("=" * 50)
    print("5. 数值实验：验证 RoPE 相对位置特性")
    print("=" * 50)
    dim = 8
    # 取两个完全相同的内容向量 (内容一致，仅位置不同)
    q_content = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    k_content = q_content.clone()

    # 场景1: 绝对位置 (i=1, j=3) → 相对距离 d = 2
    pos_i1, pos_j1 = 1, 3
    q1 = rope_embedding(q_content.unsqueeze(0), torch.tensor([pos_i1]))
    k1 = rope_embedding(k_content.unsqueeze(0), torch.tensor([pos_j1]))
    score1 = torch.dot(q1.squeeze(), k1.squeeze())

    # 场景2: 绝对位置 (i=4, j=6) → 相对距离 d = 2 (和上面相对距离一致)
    pos_i2, pos_j2 = 4, 6
    q2 = rope_embedding(q_content.unsqueeze(0), torch.tensor([pos_i2]))
    k2 = rope_embedding(k_content.unsqueeze(0), torch.tensor([pos_j2]))
    score2 = torch.dot(q2.squeeze(), k2.squeeze())

    # 场景3: 相对距离 d = 1 (不同相对位置)
    pos_i3, pos_j3 = 2, 3
    q3 = rope_embedding(q_content.unsqueeze(0), torch.tensor([pos_i3]))
    k3 = rope_embedding(k_content.unsqueeze(0), torch.tensor([pos_j3]))
    score3 = torch.dot(q3.squeeze(), k3.squeeze())

    print(f"相对距离 d=2 (pos1=1,3) 注意力点积: {score1.item():.4f}")
    print(f"相对距离 d=2 (pos2=4,6) 注意力点积: {score2.item():.4f}")
    print(f"相对距离 d=1 (pos=2,3) 注意力点积: {score3.item():.4f}")
    print("\n结论：")
    print("1. 相同**相对距离**，无论绝对位置在哪，点积结果几乎相等 → RoPE 依赖相对位置")
    print("2. 不同相对距离，点积结果明显不同 → 天然编码相对位置关系")
    print("=" * 50 + "\n")

# ===================== 主函数运行所有实验 =====================
if __name__ == "__main__":
    # 1. 测试正弦位置编码
    seq_len_test, dim_test = 10, 8
    pos_enc = sinusoidal_pos_encoding(seq_len_test, dim_test)
    print(f"1. 正弦位置编码 shape [seq_len, dim]: {pos_enc.shape}\n")

    # 2. 测试二维向量旋转
    vec_2d = torch.tensor([1.0, 0.0])
    rot_vec = rotate_2d(vec_2d, math.pi / 2)
    print(f"2. 二维向量旋转测试: 原向量 {vec_2d.tolist()}, 旋转90度后 {rot_vec.tolist()}\n")

    # 3. 测试高维RoPE
    test_x = torch.randn(seq_len_test, dim_test)
    test_pos = torch.arange(seq_len_test)
    rope_x = rope_embedding(test_x, test_pos)
    print(f"3. 高维RoPE输出 shape: {rope_x.shape}\n")

    # 4. 对比 E+pos 和 RoPE 输入方式
    compare_input_mode()

    # 5. 验证 RoPE 相对位置性质
    verify_relative_position()

    # 可视化正弦位置编码
    plt.figure(figsize=(10, 4))
    plt.imshow(pos_enc.numpy(), cmap="YlGnBu")
    plt.title("Sinusoidal Position Encoding")
    plt.xlabel("Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar()
    plt.show()