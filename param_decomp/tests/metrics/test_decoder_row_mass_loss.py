import torch

from param_decomp.metrics.decoder_column_mass import decoder_column_mass_loss
from param_decomp.metrics.decoder_row_mass import decoder_row_mass_loss


def test_dense_vs_split_rows():
    # 1 dense row of 15 ones vs 5 rows of 3 ones (the row transpose of the column test).
    b_dense = torch.zeros(1, 15)
    b_dense[0, :] = 1.0
    b_split = torch.zeros(5, 15)
    for m in range(5):
        b_split[m, m * 3 : (m + 1) * 3] = 1.0
    assert decoder_row_mass_loss(b_dense).item() == 225.0  # 15^2
    assert decoder_row_mass_loss(b_split).item() == 45.0  # 5 * 3^2


def test_merge_raises_penalty_by_2_mi_mj():
    # row0 mass 2, row1 mass 3 -> 4 + 9 = 13; merged into one row mass 5 -> 25; rise = 2*2*3 = 12.
    b_split = torch.zeros(2, 5)
    b_split[0, :2] = 1.0
    b_split[1, 2:5] = 1.0
    b_merge = torch.zeros(1, 5)
    b_merge[0, :] = 1.0
    assert decoder_row_mass_loss(b_split).item() == 13.0
    assert decoder_row_mass_loss(b_merge).item() == 25.0


def test_uses_absolute_value():
    b = torch.tensor([[-1.0, -2.0]])  # one row, mass = 3 -> 9
    assert decoder_row_mass_loss(b).item() == 9.0


def test_row_and_column_differ_on_asymmetric_B():
    # On a non-symmetric B the row and column reductions must disagree.
    b = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])  # rows: 3,4 -> 25 ; cols: 1,3,3 -> 19
    assert decoder_row_mass_loss(b).item() == 25.0
    assert decoder_column_mass_loss(b).item() == 19.0


def test_grad_to_B():
    b = torch.randn(6, 3, requires_grad=True)
    decoder_row_mass_loss(b).backward()
    assert b.grad is not None and b.grad.abs().sum() > 0
