"""Sentence VAE with the knobs studied in the paper.

Implemented levers:
  * decoder type (LSTM / CNN) and capacity
  * skip connections from z to every decoder step
  * word dropout, free bits, tied embeddings
The KL weight (annealing) is applied by the trainer, not the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        hidden_dim: int,
        latent_dim: int,
        layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            embedding.embedding_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(out_dim, latent_dim)
        self.fc_logvar = nn.Linear(out_dim, latent_dim)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        emb = self.drop(self.embedding(input_ids))
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        return self.fc_mu(h), self.fc_logvar(h)


class _DecoderBase(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        hidden_dim: int,
        vocab_size: int,
        latent_dim: int,
        layers: int,
        dropout: float,
        skip_connection: bool,
        tie_embeddings: bool,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.layers = layers
        self.skip_connection = skip_connection
        self.drop = nn.Dropout(dropout)
        if tie_embeddings and embedding.embedding_dim == hidden_dim:
            self.out = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.out.weight = embedding.weight
        else:
            self.out = nn.Linear(hidden_dim, vocab_size)

    def _decoder_input(self, z: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.drop(self.embedding(input_ids))
        if self.skip_connection:
            emb = torch.cat([emb, z.unsqueeze(1).expand(-1, emb.size(1), -1)], dim=-1)
        return emb


class LSTMDecoder(_DecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_dim = self.embedding.embedding_dim + (self.latent_dim if self.skip_connection else 0)
        self.lstm = nn.LSTM(
            in_dim,
            self.hidden_dim,
            num_layers=self.layers,
            batch_first=True,
            dropout=0.0,
        )
        self.z2h = nn.Linear(self.latent_dim, self.hidden_dim * self.layers)
        self.z2c = nn.Linear(self.latent_dim, self.hidden_dim * self.layers)

    def forward(
        self, z: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self._decoder_input(z, input_ids)
        batch = z.size(0)
        h0 = torch.tanh(self.z2h(z)).view(batch, self.layers, self.hidden_dim)
        c0 = torch.tanh(self.z2c(z)).view(batch, self.layers, self.hidden_dim)
        h0 = h0.transpose(0, 1).contiguous()
        c0 = c0.transpose(0, 1).contiguous()
        if lengths is not None:
            packed = pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed, (h0, c0))
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        else:
            out, _ = self.lstm(emb, (h0, c0))
        return self.out(out)


class CNNDecoder(_DecoderBase):
    def __init__(self, *args, kernel_size: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_dim = self.embedding.embedding_dim + (self.latent_dim if self.skip_connection else 0)
        self.conv1 = nn.Conv1d(in_dim, self.hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(
            self.hidden_dim, self.hidden_dim, kernel_size, padding=kernel_size // 2
        )

    def forward(
        self, z: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        del lengths  # convolutions are length-preserving; padding is masked in the loss
        x = self._decoder_input(z, input_ids).transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return self.out(x.transpose(1, 2))


class SentenceVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        enc_layers: int = 1,
        dec_layers: int = 1,
        dropout: float = 0.3,
        decoder: str = "lstm",
        bidirectional: bool = True,
        skip_connection: bool = False,
        tie_embeddings: bool = True,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 3,
        word_dropout: float = 0.0,
        free_bits: float = 0.0,
    ) -> None:
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad_id, bos_id, eos_id, unk_id
        self.word_dropout = word_dropout
        self.free_bits = free_bits
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = Encoder(
            self.embedding, hidden_dim, latent_dim, enc_layers, dropout, bidirectional
        )
        decoder_cls = {"lstm": LSTMDecoder, "cnn": CNNDecoder}[decoder]
        self.decoder = decoder_cls(
            self.embedding,
            hidden_dim,
            vocab_size,
            latent_dim,
            dec_layers,
            dropout,
            skip_connection,
            tie_embeddings,
        )

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        return self.encoder(input_ids, lengths)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _drop_words(self, decoder_input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.word_dropout <= 0:
            return decoder_input
        mask = torch.rand(decoder_input.shape, device=decoder_input.device) < self.word_dropout
        special = (
            (decoder_input == self.pad_id)
            | (decoder_input == self.bos_id)
            | (decoder_input == self.eos_id)
        )
        return decoder_input.masked_fill(mask & ~special, self.unk_id)

    def forward(
        self, input_ids: torch.Tensor, lengths: torch.Tensor, beta: float = 1.0
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(input_ids, lengths)
        z = self.reparameterise(mu, logvar)

        decoder_input = self._drop_words(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        decoder_lengths = (lengths - 1).clamp(min=1)
        logits = self.decoder(z, decoder_input, decoder_lengths)

        recon_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_id,
            reduction="sum",
        )
        kl_raw = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_elem = kl_raw.clamp(min=self.free_bits) if self.free_bits > 0 else kl_raw
        kl = kl_elem.sum(1).mean()

        batch = input_ids.size(0)
        recon = recon_sum / batch
        loss = recon + beta * kl

        n_tokens = targets.ne(self.pad_id).sum().clamp(min=1)
        return {
            "loss": loss,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "kl_per_dim": kl_raw.mean(0).detach(),
            "nll_per_token": (recon_sum / n_tokens).detach(),
            "n_tokens": n_tokens.detach(),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def reconstruct(
        self, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        mu, _ = self.encode(input_ids, lengths)
        logits = self.decoder(mu, input_ids[:, :-1], (lengths - 1).clamp(min=1))
        return logits.argmax(-1)
