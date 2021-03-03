from typing import List

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify

Default_Shape = (256, 512)


def pytorch2onnx(model: nn.Module, output_file: str, input_shape: List[int] = Default_Shape):
    # Export to ONNX
    inputs = torch.ones(*input_shape)
    torch.onnx.export(model.eval(), inputs, output_file)

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
        '--dims',
        nargs='+',
        type=int,
        default=Default_Shape,
        help=f'dimensions of model input separated by space. Default: {"x".join(str(i) for i in Default_Shape)}',
    )

    args = parser.parse_args()

    model = torch.load(args.model_file)

    pytorch2onnx(model, args.output_file, args.dims)
