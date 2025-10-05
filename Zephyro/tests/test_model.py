import pytest

torch = pytest.importorskip("torch")

from aqi.model_transformer import AQITransformer, ModelConfig


def test_model_forward_shape():
    config = ModelConfig(input_size=5, forecast_horizon=7, d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
    model = AQITransformer(config)
    X = torch.randn(4, 10, 5)
    out = model(X)
    assert out.shape == (4, 7)
