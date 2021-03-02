from typing import Tuple

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify


def pytorch2onnx(model: nn.Module, input_shape: Tuple[int, ...], output_file: str):
    # Export to ONNX
    inputs = torch.ones(*input_shape)
    torch.onnx.export(model, inputs, output_file)

    # Simplify ONNX
    model = onnx.load(output_file)
    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, output_file)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog='PyTorch to TensorRT converter',
        description='Converts PyTorch model to TensorRT compatible object',
    )

    parser.add_argument(
        'model_file',
        help='path to file with model weights',
    )
    parser.add_argument(
        'output_file',
        help='path to resulting file',
    )
    parser.add_argument(
        'model_module',
        help='path to module with model class definition',
    )
    parser.add_argument(
        'model_class',
        help='qualified name of model class',
    )
    parser.add_argument(
        '--dims',
        nargs='+',
        type=int,
        default=(256, 512),
        help='dimensions of model input separated by space. Default: 256x512',
    )

    args = parser.parse_args()

    module = __import__(args.model_module)
    Model: type = getattr(module, args.model_class)
    model_dict = torch.load(args.model_file)
    model: nn.Module = Model().eval()
    model.load_state_dict(model_dict)

    pytorch2onnx(model, args.dims, args.output_file)
