import pytest
import torch
from model.lgsp import LocalGlobalSpatialFusion


@pytest.mark.parametrize('mode', ['baseline', 'local', 'frequency', 'both'])
def test_fusion_formula_and_disable(mode):
    m = LocalGlobalSpatialFusion(mode, diagnostics=True)
    x = torch.randn(2, 3, 32, 16)
    expected = x.float()
    for branch in (m.local, m.frequency):
        if branch is not None:
            expected = expected + branch(x)
    torch.testing.assert_close(m(x), expected)
    assert m(x).dtype == x.dtype
    if mode == 'baseline':
        assert m(x) is x
    else:
        assert m.last_diagnostics
    for branch in (m.local, m.frequency):
        if branch is not None:
            branch.alpha.data.zero_()
    assert torch.equal(m(x), x)
