import torch

from bcvae.generation import decode_from_z
from bcvae.models import SentenceVAE


def _batch():
    ids = torch.tensor([[1, 5, 6, 2, 0], [1, 7, 8, 9, 2]])
    lengths = torch.tensor([4, 5])
    return ids, lengths


def test_forward_backward():
    torch.manual_seed(0)
    model = SentenceVAE(
        vocab_size=64,
        emb_dim=16,
        hidden_dim=32,
        latent_dim=4,
        dropout=0.0,
        word_dropout=0.2,
        free_bits=0.0,
    )
    ids, lengths = _batch()
    out = model(ids, lengths, beta=0.5)
    assert torch.isfinite(out["loss"])
    assert out["kl_per_dim"].shape == (4,)
    out["loss"].backward()
    assert model.encoder.fc_mu.weight.grad is not None


def test_free_bits_lower_bounds_kl():
    torch.manual_seed(0)
    model = SentenceVAE(
        vocab_size=64,
        emb_dim=16,
        hidden_dim=32,
        latent_dim=4,
        dropout=0.0,
        free_bits=0.5,
    )
    ids, lengths = _batch()
    out = model(ids, lengths)
    assert float(out["kl"]) >= 0.5 * 4 - 1e-4


def test_cnn_decoder_and_sampling():
    model = SentenceVAE(
        vocab_size=64,
        emb_dim=16,
        hidden_dim=16,
        latent_dim=4,
        decoder="cnn",
        dropout=0.0,
        skip_connection=True,
    )
    ids, lengths = _batch()
    out = model(ids, lengths)
    assert out["loss"].shape == ()

    z = torch.randn(3, 4)
    decoded = decode_from_z(model, z, bos_id=1, eos_id=2, max_len=8)
    assert decoded.shape[0] == 3
    assert decoded[0, 0].item() == 1
