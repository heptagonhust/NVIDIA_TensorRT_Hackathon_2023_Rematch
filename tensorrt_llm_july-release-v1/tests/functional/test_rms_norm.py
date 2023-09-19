import unittest

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Parameter, Tensor
from tensorrt_llm._utils import torch_to_numpy
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.randn((dim), dtype=torch.float32, device="cuda")

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')
        torch.manual_seed(42)

    @parameterized.expand([['float16'], ['float32']])
    def test_layer_norm_plugin(self, dtype):
        # test data
        hidden_size = 4096
        x_data = torch.randn((8, 128, hidden_size),
                             dtype=torch.float64,
                             device="cuda")  # torch.Size([8, 128, 1024])
        weight = torch.randn((hidden_size), dtype=torch.float64, device="cuda")
        bias = torch.randn((hidden_size), dtype=torch.float64, device="cuda")
        eps = 1e-5
        m = RMSNorm(hidden_size, eps)
        # m = torch.nn.LayerNorm(hidden_size,
        #                        eps=eps,
        #                        dtype=torch.float64,
        #                        device="cuda")
        # m.weight = torch.nn.Parameter(weight)
        # m.bias = torch.nn.Parameter(bias)

        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        m.to(tensorrt_llm._utils.str_dtype_to_torch(dtype))
        x_data = x_data.to(tensorrt_llm._utils.str_dtype_to_torch(dtype))

        gamma_data = m.weight.detach().cpu()
        # beta_data = m.bias.detach().cpu()

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_RMSnorm_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            weight = Parameter(torch_to_numpy(gamma_data.cpu())).value
            # bias = Parameter(torch_to_numpy(beta_data.cpu())).value

            output = tensorrt_llm.functional.rms_norm(x, hidden_size, weight,
                                                         eps).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16'),
                                bf16=(dtype == 'bfloat16')))
        assert build_engine is not None, "Build engine failed"
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.cpu()})

        # compare diff
        dtype_atol = {"float16": 2e-2, "float32": 2e-6, "bfloat16": 8e-2}
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'].to(torch.float32),
                                   atol=dtype_atol[dtype])
        
if __name__ == "__main__":
    unittest.main()

