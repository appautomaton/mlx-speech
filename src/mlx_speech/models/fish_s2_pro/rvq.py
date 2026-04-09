import mlx.nn as nn
import mlx.core as mx


class RVQDecoder(nn.Module):
    def __init__(
        self,
        num_codebooks: int = 10,
        codebook_size: int = 4096,
        dim: int = 1024,
        sample_rate: int = 22050,
        frame_rate: int = 21,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.hop_length = sample_rate // frame_rate

        self.codebooks = [
            nn.Embedding(codebook_size, dim) for _ in range(num_codebooks)
        ]

        self.to_waveform = nn.Linear(dim, self.hop_length)

    def decode(self, codes: mx.array) -> mx.array:
        batch, time, num_cb = codes.shape

        embeddings = []
        for i in range(self.num_codebooks):
            indices = codes[:, :, i]
            cb_emb = self.codebooks[i](indices)
            embeddings.append(cb_emb)

        h = sum(embeddings)

        h = self.to_waveform(h)

        audio = h.reshape(batch, -1)

        return audio
