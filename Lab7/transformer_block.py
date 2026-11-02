"""LAB 07, Bài 17–18: Transformer Encoder block tự ghép bằng PyTorch."""

from __future__ import annotations

import math

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Một Encoder block Post-LN với multi-head self-attention và FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model và d_ff phải là số nguyên dương")
        if n_heads <= 0:
            raise ValueError("n_heads phải là số nguyên dương")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong khoảng [0, 1)")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Bốn phép chiếu học được dùng chung cho mọi vị trí trong sequence.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN áp dụng độc lập trên chiều features của từng token.
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """Nhận X có shape (B, T, d_model), trả output cùng shape.

        Khi return_debug=True, trả thêm metadata shape và sai số softmax,
        không giữ các tensor trung gian lớn trong dictionary debug.
        """
        if x.ndim != 3:
            raise ValueError("x phải có shape (B, T, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"chiều cuối của x phải là d_model={self.d_model}, nhận {x.shape[-1]}"
            )
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("batch size và sequence length phải lớn hơn 0")
        if return_debug and return_attention:
            raise ValueError("chỉ chọn một trong return_debug và return_attention")

        batch_size, sequence_length, _ = x.shape
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask phải có shape (B, T)")
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            if not torch.all(attention_mask.any(dim=1)):
                raise ValueError("mỗi sequence cần ít nhất một key không phải padding")

        # Q, K, V trước khi tách head: mỗi tensor có shape (B, T, d_model).
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query_projected_shape = tuple(query.shape)
        key_projected_shape = tuple(key.shape)
        value_projected_shape = tuple(value.shape)

        # Tách chiều cuối thành (n_heads, d_k), rồi chuyển head lên trước T.
        query = query.reshape(batch_size, sequence_length, self.n_heads, self.d_k)
        key = key.reshape(batch_size, sequence_length, self.n_heads, self.d_k)
        value = value.reshape(batch_size, sequence_length, self.n_heads, self.d_k)
        query = query.transpose(1, 2)  # (B, n_heads, T, d_k)
        key = key.transpose(1, 2)      # (B, n_heads, T, d_k)
        value = value.transpose(1, 2)  # (B, n_heads, T, d_k)

        # Attention scores có shape (B, n_heads, T_query, T_key).
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attention_mask is not None:
            # Mask các key PAD trước softmax để mô hình không attention vào padding.
            scores = scores.masked_fill(
                ~attention_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )
        attention_weights = torch.softmax(scores, dim=-1)
        attention_per_head = torch.matmul(attention_weights, value)

        # Ghép các head về (B, T, d_model) trước output projection.
        concatenated_heads = attention_per_head.transpose(1, 2).contiguous()
        concatenated_heads = concatenated_heads.reshape(
            batch_size, sequence_length, self.d_model
        )
        attention_output = self.out_proj(concatenated_heads)

        # Post-LN: chuẩn hóa sau residual của attention.
        residual1_input = x + self.dropout(attention_output)
        x1 = self.norm1(residual1_input)

        # FFN đổi chiều features d_model -> d_ff -> d_model cho từng token.
        ffn_hidden = torch.relu(self.ffn_in(x1))
        ffn_output = self.ffn_out(ffn_hidden)

        # Post-LN lần hai: residual từ x1 cộng với FFN output.
        residual2_input = x1 + self.dropout(ffn_output)
        output = self.norm2(residual2_input)

        if output.shape != x.shape:
            raise RuntimeError("TransformerBlock phải giữ nguyên shape đầu vào")

        if return_attention:
            return output, attention_weights
        if not return_debug:
            return output

        row_sums = attention_weights.sum(dim=-1)
        debug = {
            "Input X": tuple(x.shape),
            "Q before splitting heads": query_projected_shape,
            "K before splitting heads": key_projected_shape,
            "V before splitting heads": value_projected_shape,
            "Q after splitting heads": tuple(query.shape),
            "K after splitting heads": tuple(key.shape),
            "V after splitting heads": tuple(value.shape),
            "Attention scores": tuple(scores.shape),
            "Attention weights": tuple(attention_weights.shape),
            "Attention output per head": tuple(attention_per_head.shape),
            "Concatenated heads": tuple(concatenated_heads.shape),
            "Output projection": tuple(attention_output.shape),
            "Residual + LayerNorm 1": tuple(x1.shape),
            "FFN hidden": tuple(ffn_hidden.shape),
            "FFN output": tuple(ffn_output.shape),
            "Residual + LayerNorm 2": tuple(output.shape),
            "Final output": tuple(output.shape),
            "attention_row_sum_max_error": float((row_sums - 1.0).abs().max().detach().cpu()),
            "attention_rows_sum_to_one": bool(
                torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6)
            ),
        }
        return output, debug


def parameter_counts(model: TransformerBlock) -> dict[str, int]:
    """Đếm tham số trainable theo attention, FFN và LayerNorm."""
    attention_modules = (model.q_proj, model.k_proj, model.v_proj, model.out_proj)
    ffn_modules = (model.ffn_in, model.ffn_out)
    norm_modules = (model.norm1, model.norm2)

    def count(modules) -> int:
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    counts = {
        "attention": count(attention_modules),
        "FFN": count(ffn_modules),
        "LayerNorm": count(norm_modules),
    }
    counts["total"] = sum(counts.values())
    return counts


def verify_main_shapes() -> tuple[TransformerBlock, torch.Tensor, dict[str, object]]:
    """Chạy đúng cấu hình shape verification trong W7.pdf."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, sequence_length = 8, 100
    d_model, n_heads, d_ff = 128, 4, 512

    # d_ff=512 là cấu hình thử nghiệm bổ sung; W7.pdf yêu cầu B,T,d_model,n_heads.
    model = TransformerBlock(d_model, n_heads, d_ff).to(device)
    x = torch.randn(batch_size, sequence_length, d_model, device=device)
    model.eval()
    with torch.no_grad():
        output, debug = model(x, return_debug=True)

    print("Bài 18 — Tensor shapes thực tế")
    for name, value in debug.items():
        if isinstance(value, tuple):
            print(f"{name}: {value}")
    print(f"Attention weight row-sum max error: {debug['attention_row_sum_max_error']:.3e}")
    print(f"Attention weights sum to one: {debug['attention_rows_sum_to_one']}")
    assert output.shape == (8, 100, 128)
    assert debug["Q after splitting heads"] == (8, 4, 100, 32)
    assert debug["K after splitting heads"] == (8, 4, 100, 32)
    assert debug["V after splitting heads"] == (8, 4, 100, 32)
    assert debug["Attention scores"] == (8, 4, 100, 100)
    assert debug["Attention output per head"] == (8, 4, 100, 32)
    assert debug["FFN hidden"] == (8, 100, 512)
    assert debug["Final output"] == (8, 100, 128)
    assert debug["attention_rows_sum_to_one"]
    return model, x, debug


def test_residual_connections() -> None:
    """Dùng hooks để xác nhận hai LayerNorm nhận đúng residual sums."""
    torch.manual_seed(7)
    model = TransformerBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
    model.eval()
    x = torch.randn(2, 5, 16)
    captured: dict[str, torch.Tensor] = {}

    handles = [
        model.out_proj.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("attention_output", output.detach())
        ),
        model.norm1.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("norm1_input", inputs[0].detach())
        ),
        model.norm1.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("norm1_output", output.detach())
        ),
        model.ffn_out.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("ffn_output", output.detach())
        ),
        model.norm2.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("norm2_input", inputs[0].detach())
        ),
    ]
    with torch.no_grad():
        model(x)
    for handle in handles:
        handle.remove()

    torch.testing.assert_close(captured["norm1_input"], x + captured["attention_output"])
    torch.testing.assert_close(captured["norm2_input"], captured["norm1_output"] + captured["ffn_output"])


def run_tests() -> None:
    """Chạy shape, softmax, residual, multi-head, lỗi dimension và backward tests."""
    print("\nBài 17 — Unit tests")
    _, _, debug = verify_main_shapes()
    print("PASS: input/output và intermediate shapes theo cấu hình đề bài")
    print("PASS: Q/K/V tách head, scores và weights đúng shapes")
    print("PASS: attention weights cộng bằng 1 trên chiều key")

    test_residual_connections()
    print("PASS: hai residual connections đưa đúng tensor vào LayerNorm")

    masked_model = TransformerBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    _, attention = masked_model(
        torch.randn(2, 4, 16), attention_mask=mask, return_attention=True
    )
    assert torch.all(attention.masked_select(~mask[:, None, None, :]) == 0)
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)))
    print("PASS: padding keys bị mask và API trả attention weights hoạt động")

    for n_heads in (1, 2, 4, 8):
        model = TransformerBlock(d_model=128, n_heads=n_heads, d_ff=512, dropout=0.0)
        output = model(torch.randn(2, 7, 128))
        assert output.shape == (2, 7, 128)
    print("PASS: n_heads = 1, 2, 4, 8; output shape được giữ nguyên")

    try:
        TransformerBlock(d_model=130, n_heads=4, d_ff=512)
    except ValueError as error:
        assert "chia hết" in str(error)
        print(f"PASS: invalid dimensions báo lỗi rõ ràng: {error}")
    else:
        raise AssertionError("d_model=130, n_heads=4 phải gây ValueError")

    torch.manual_seed(11)
    model = TransformerBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
    x = torch.randn(3, 6, 16, requires_grad=True)
    output = model(x)
    loss = output.square().mean()
    loss.backward()
    important_parameters = (
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
        "ffn_in.weight", "ffn_out.weight", "norm1.weight", "norm2.weight",
    )
    for name, parameter in model.named_parameters():
        if name in important_parameters:
            assert parameter.grad is not None, f"missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    print("PASS: backward tạo gradient hữu hạn cho attention, FFN, LayerNorm và input")
    print("PASS: forward, loss và gradients không có NaN/Inf")


def main() -> None:
    """In cấu hình, parameter counts, shapes và kết quả kiểm thử."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("LAB 07 — Transformer Encoder block from scratch")
    print(f"PyTorch: {torch.__version__}; device: {device}")
    print("Built-in Transformer / MultiheadAttention / pretrained model: không dùng")

    model = TransformerBlock(d_model=128, n_heads=4, d_ff=512)
    counts = parameter_counts(model)
    print("\nTrainable parameter counts:")
    for name, number in counts.items():
        print(f"{name}: {number:,}")

    run_tests()
    print("\nTất cả kiểm thử đã đạt.")


if __name__ == "__main__":
    main()
