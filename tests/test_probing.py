import torch

from bcvae.probing import fit_linear_probe


def test_probe_reports_balanced_metrics():
    torch.manual_seed(0)
    # Three well-separated classes with a skewed class distribution.
    centres = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    )
    sizes = [180, 30, 20]
    z = torch.cat(
        [
            centres[cls] + 0.2 * torch.randn(size, 3)
            for cls, size in enumerate(sizes)
        ]
    )
    y = torch.cat(
        [torch.full((size,), cls, dtype=torch.long) for cls, size in enumerate(sizes)]
    )

    result = fit_linear_probe(z, y, 3, epochs=200, seed=0)

    assert result["accuracy"] > 0.9
    assert result["balanced_accuracy"] > 0.9
    assert result["macro_f1"] > 0.9
    assert 0.0 <= result["majority"] <= 1.0
    assert result["n"] == len(y)
    assert result["n_val"] > 0
