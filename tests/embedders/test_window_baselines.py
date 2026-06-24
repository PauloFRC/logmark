import pytest
import torch

from logmark.embedders.window_baselines import (
    CountVectorEncoder,
    PoolingEmbeddingEncoder,
    MeanPoolingEncoder,
    AttentionPoolingEncoder
)
from logmark.embedders.window_base import build_encoder

class TestCountVectorEncoder:
    def test_output_dim(self):
        encoder = CountVectorEncoder(vocab_size=10)
        assert encoder.output_dim == 10

    def test_forward_basic_counting(self):
        vocab_size = 5
        encoder = CountVectorEncoder(vocab_size=vocab_size, normalize="none")
        padded_input = torch.tensor([
            [1, 1, 2, 0],
            [3, 4, 3, 3]
        ])
        mask = torch.tensor([
            [True, True, True, False],
            [True, True, True, True]
        ])
        
        output = encoder(padded_input, mask)
        assert output.shape == (2, vocab_size)
        
        expected_row_0 = torch.tensor([0.0, 2.0, 1.0, 0.0, 0.0])
        assert torch.allclose(output[0], expected_row_0)
        
        expected_row_1 = torch.tensor([0.0, 0.0, 0.0, 3.0, 1.0])
        assert torch.allclose(output[1], expected_row_1)

    def test_forward_l1_normalization(self):
        vocab_size = 4
        encoder = CountVectorEncoder(vocab_size=vocab_size, normalize="l1")
        
        padded_input = torch.tensor([
            [1, 1, 2, 0],
        ])
        mask = torch.tensor([
            [True, True, True, False],
        ])
        
        output = encoder(padded_input, mask)

        expected = torch.tensor([0.0, 2/3, 1/3, 0.0])
        assert torch.allclose(output[0], expected, atol=1e-5)

    def test_all_masked_batch(self):
        vocab_size = 3
        encoder = CountVectorEncoder(vocab_size=vocab_size)
        
        padded_input = torch.tensor([[0, 0, 0]])
        mask = torch.tensor([[False, False, False]])
        
        output = encoder(padded_input, mask)
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(output[0], expected)


class TestPoolingEmbeddingEncoder:
    def test_mean_pooling(self):
        vocab_size = 10
        dim = 8
        encoder = PoolingEmbeddingEncoder(vocab_size=vocab_size, dim=dim, aggregation="mean")
        
        padded_input = torch.tensor([
            [1, 2, 0]
        ])
        mask = torch.tensor([
            [True, True, False]
        ])
        
        output = encoder(padded_input, mask)
        assert output.shape == (1, dim)
        
        emb_weights = encoder.embedding.weight.data
        expected = (emb_weights[1] + emb_weights[2]) / 2.0
        assert torch.allclose(output[0], expected, atol=1e-5)

    def test_max_pooling(self):
        vocab_size = 10
        dim = 8
        encoder = PoolingEmbeddingEncoder(vocab_size=vocab_size, dim=dim, aggregation="max")
        
        padded_input = torch.tensor([
            [1, 2, 0]
        ])
        mask = torch.tensor([
            [True, True, False]
        ])
        
        output = encoder(padded_input, mask)
        
        emb_weights = encoder.embedding.weight.data
        expected = torch.max(torch.stack([emb_weights[1], emb_weights[2]]), dim=0).values
        assert torch.allclose(output[0], expected, atol=1e-5)

    def test_sum_pooling(self):
        vocab_size = 10
        dim = 8
        encoder = PoolingEmbeddingEncoder(vocab_size=vocab_size, dim=dim, aggregation="sum")
        
        padded_input = torch.tensor([
            [1, 2, 0]
        ])
        mask = torch.tensor([
            [True, True, False]
        ])
        
        output = encoder(padded_input, mask)
        
        emb_weights = encoder.embedding.weight.data
        expected = emb_weights[1] + emb_weights[2]
        assert torch.allclose(output[0], expected, atol=1e-5)


class TestAttentionPoolingEncoder:
    def test_output_shape_and_masking(self):
        vocab_size = 10
        dim = 8
        encoder = AttentionPoolingEncoder(vocab_size=vocab_size, dim=dim)
        
        padded_input = torch.tensor([
            [1, 2, 3],
            [1, 0, 0]
        ])
        mask = torch.tensor([
            [True, True, True],
            [False, False, False] # all masked
        ])
        
        output = encoder(padded_input, mask)
        
        assert output.shape == (2, dim)
        
        assert torch.allclose(output[1], torch.zeros(dim), atol=1e-5)

class TestWindowEncoderRegistry:
    def test_build_encoders(self):
        count_enc = build_encoder("count", vocab_size=50)
        assert isinstance(count_enc, CountVectorEncoder)
        
        mean_enc = build_encoder("mean_pool", vocab_size=50, dim=16)
        assert isinstance(mean_enc, MeanPoolingEncoder)
        
        attn_enc = build_encoder("attention_pool", vocab_size=50, dim=16)
        assert isinstance(attn_enc, AttentionPoolingEncoder)

    def test_unknown_encoder(self):
        with pytest.raises(ValueError, match="Unknown encoder"):
            build_encoder("fake_encoder", vocab_size=50)
