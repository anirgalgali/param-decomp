import torch

from param_decomp.metrics.decoder_column_mass import decoder_column_mass_loss


def test_dense_vs_split_matches_design_table():
    # 1 dense column of 15 ones vs 5 columns of 3 ones (design §8 table).
    b_dense = torch.zeros(15, 1)
    b_dense[:, 0] = 1.0
    b_split = torch.zeros(15, 5)
    for k in range(5):
        b_split[k * 3 : (k + 1) * 3, k] = 1.0
    assert decoder_column_mass_loss(b_dense).item() == 225.0  # 15^2
    assert decoder_column_mass_loss(b_split).item() == 45.0  # 5 * 3^2


def test_merge_raises_penalty_by_2_mi_mj():
    # col0 mass 2, col1 mass 3 -> 4 + 9 = 13; merged mass 5 -> 25; rise = 2*2*3 = 12.
    b_split = torch.zeros(5, 2)
    b_split[:2, 0] = 1.0
    b_split[2:5, 1] = 1.0
    b_merge = torch.zeros(5, 1)
    b_merge[:, 0] = 1.0
    assert decoder_column_mass_loss(b_split).item() == 13.0
    assert decoder_column_mass_loss(b_merge).item() == 25.0


def test_uses_absolute_value():
    b = torch.tensor([[-1.0], [-2.0]])  # mass = 3 -> 9
    assert decoder_column_mass_loss(b).item() == 9.0


def test_grad_to_B():
    b = torch.randn(6, 3, requires_grad=True)
    decoder_column_mass_loss(b).backward()
    assert b.grad is not None and b.grad.abs().sum() > 0
