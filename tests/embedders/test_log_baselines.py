import pytest
import torch

from logmark.embedders.log_baselines import (
    TokenPoolingEncoder,
    MeanTokenPoolingEncoder,
    TransformerLogEncoder,
    HAS_TRANSFORMERS
)
from logmark.embedders.log_base import build_single_encoder

class TestTokenPoolingEncoder:
    def test_output_dim(self):
        encoder = TokenPoolingEncoder(vocab_size=100, dim=32)
        assert encoder.output_dim == 32

    def test_forward_mean_pooling(self):
        vocab_size = 100
        dim = 16
        encoder = TokenPoolingEncoder(vocab_size=vocab_size, dim=dim, aggregation="mean")
        
        padded_tokens = torch.tensor([
            [1, 2, 0],
            [3, 4, 5]
        ])
        
        mask = torch.tensor([
            [True, True, False],
            [True, True, True]
        ])
        
        output = encoder(padded_tokens, mask)
        
        assert output.shape == (2, dim)
        
        embedding_matrix = encoder.embedding.weight.data
        expected_item1 = (embedding_matrix[1] + embedding_matrix[2]) / 2.0
        assert torch.allclose(output[0], expected_item1, atol=1e-5)
        
        expected_item2 = (embedding_matrix[3] + embedding_matrix[4] + embedding_matrix[5]) / 3.0
        assert torch.allclose(output[1], expected_item2, atol=1e-5)

    def test_forward_max_pooling(self):
        vocab_size = 100
        dim = 16
        encoder = TokenPoolingEncoder(vocab_size=vocab_size, dim=dim, aggregation="max")
        
        padded_tokens = torch.tensor([
            [1, 2, 0],
            [0, 0, 0]
        ])
        
        mask = torch.tensor([
            [True, True, False],
            [False, False, False]
        ])
        
        output = encoder(padded_tokens, mask)
        
        assert output.shape == (2, dim)
        
        embedding_matrix = encoder.embedding.weight.data
        expected_item1 = torch.max(
            torch.stack([embedding_matrix[1], embedding_matrix[2]]), dim=0
        ).values
        assert torch.allclose(output[0], expected_item1, atol=1e-5)
        
        expected_item2 = torch.zeros(dim)
        assert torch.allclose(output[1], expected_item2, atol=1e-5)

    def test_oov_handling(self):
        vocab_size = 50
        dim = 8
        encoder = TokenPoolingEncoder(vocab_size=vocab_size, dim=dim, aggregation="mean")
        
        padded_tokens = torch.tensor([
            [50, 10, 0],
            [100, 200, 0]
        ])
        mask = torch.tensor([
            [True, True, False],
            [True, True, False]
        ])
        
        output = encoder(padded_tokens, mask)
        
        embedding_matrix = encoder.embedding.weight.data
        expected_item1 = (embedding_matrix[49] + embedding_matrix[10]) / 2.0
        assert torch.allclose(output[0], expected_item1, atol=1e-5)
        
        expected_item2 = (embedding_matrix[49] + embedding_matrix[49]) / 2.0
        assert torch.allclose(output[1], expected_item2, atol=1e-5)

class TestMeanTokenPoolingEncoder:
    def test_builder_registration(self):
        encoder = build_single_encoder("token_mean_pool", vocab_size=100, dim=32)
        assert isinstance(encoder, MeanTokenPoolingEncoder)
        assert encoder.output_dim == 32
        assert encoder.aggregation == "mean"

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers library is not installed")
class TestTransformerLogEncoder:
    def test_initialization_and_frozen_weights(self):
        model_name = "hf-internal-testing/tiny-random-bert"
        
        encoder_frozen = TransformerLogEncoder(model_name=model_name, fine_tune=False)
        for param in encoder_frozen.transformer.parameters():
            assert not param.requires_grad
            
        encoder_unfrozen = TransformerLogEncoder(model_name=model_name, fine_tune=True)
        assert any(param.requires_grad for param in encoder_unfrozen.transformer.parameters())

    def test_forward_pass_shape(self):
        model_name = "hf-internal-testing/tiny-random-bert"
        encoder = TransformerLogEncoder(model_name=model_name, fine_tune=False)
        
        padded_tokens = torch.tensor([
            [101, 200, 300, 102, 0],
            [101, 400, 102, 0, 0]
        ])
        mask = torch.tensor([
            [True, True, True, True, False],
            [True, True, True, False, False]
        ])
        
        output = encoder(padded_tokens, mask)
        
        assert output.shape == (2, encoder.output_dim)
        assert encoder.output_dim == encoder.config.hidden_size
