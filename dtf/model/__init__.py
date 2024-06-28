# from .transformer import Decoder as Transformer
from .deep_tensor_net import (
    Deep_Tensor_Net,
    Deep_Tensor_Net_conv,
    Deep_Tensor_Net_conv2d,
)

from argparse import ArgumentParser, Namespace


def get_model_parser(model_name) -> ArgumentParser:
    parser = ArgumentParser()

    if model_name in [
        "Deep_Tensor_Net",
        "Deep_Tensor_Net_conv",
        "Deep_Tensor_Net_conv2d",
    ]:
        parser.add_argument("--tensor_width", type=int, default=None)
        parser.add_argument("--model_rank", type=int, default=0, nargs="+")
        parser.add_argument(
            "--decomposition_type", type=str, default="FC_embed0_customL2"
        )
        parser.add_argument("--init_scale", type=float, default=1.0)
        parser.add_argument("--layer_type", type=str, default=None, choices=["FC"])

    else:
        raise NotImplementedError

    return parser
