import numpy as np
import torch
from nn.model import ViTPose
import os
from .backend import Backend
from constants import CACHE_DIR
from io import BytesIO


class TorchBackend(Backend):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = ViTPose(self.config)
        self.device = None

    def load(self, path: str, device: str | torch.device = None, **kargs):
        super().load(path, **kargs)
        assert path.endswith('.pth') or path.endswith('pt'), \
            f"File {path} is not a .pth file"
        
        checkpoint = torch.load(path, map_location='cpu')
        checkpoint = checkpoint['state_dict'] if 'state_dict' in checkpoint else \
            checkpoint

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() \
                else 'cpu'
            )
        elif isinstance(device, str):
            device = torch.device(device)

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.model.to(device)
        self.device = device
        return self

    def __call__(self, im):
        im = torch.from_numpy(im).to(self.device)
        return self.model(im).to('cpu').detach().numpy()
    
    def __str__(self):
        return f"PyTorch model with {self.model.num_parameters} parameters"
    
    def get_input_shape(self):
        H, W = self.config["backbone"]["img_size"]
        return (1, 3, H, W)
    
    def export_coreml(self, path: str, **kargs):
        path.endswith('.mlmodel') or path.endswith('.mlpackage'), \
            f"File {path} is not a .mlmodel or .mlpackage file"
        import coremltools as ct
        
        B, C, H, W = self.get_input_shape()
        model = self.model
        model.eval()
        inputs = torch.randn(B, C, H, W)
        traced_model = torch.jit.trace(model, inputs)
        input_shape = ct.Shape(shape=
            (ct.RangeDim(lower_bound=1, upper_bound=10, default=1), C, H, W)
        )
        coreml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="input", shape=input_shape)],
            outputs=[ct.TensorType(name="output")],
        )
        coreml_model.save(path)


    def export_onnx(self, path: str | BytesIO, opset: int = 11, verbose: bool = False):
        """
        Export model to ONNX format
        model (ViTPose): model to export
        inputs (torch.Tensor): input tensor
        output_path (str): path to save the model
        opset (int): ONNX opset version
        verbose (bool): verbose mode
        """

        assert path.endswith('.onnx'), f"File {path} is not a .onnx file"
        dynamic_axes = {'input_0': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}}
        input_names = ["input_0"]
        output_names = ["output_0"]
        B, C, H, W = self.get_input_shape()
        inputs = torch.randn(B, C, H, W).to(self.device)
        torch.onnx.export(
            self.model,
            inputs,
            path,
            export_params=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes
        )

    def export_engine(self,
        output_path,
        half: bool = False,
        dynamic: bool = True,
        workspace: int =4,
        verbose: bool = False,
        **kargs
    ):  
        B, C, H, W = self.get_input_shape()
        inputs = torch.randn(B, C, H, W).to(self.device)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            self.export_onnx(f.name, verbose=verbose, **kargs)
            f.seek(0)
            export_engine(f.name, inputs, output_path, half=half,
                          dynamic=dynamic, workspace=workspace, verbose=verbose)


def export_engine(
    onnx: str,
    im: torch.tensor,
    file: str,
    half: bool = False,
    dynamic: bool = True,
    workspace: int =4,
    verbose: bool = False,
    prefix='Tensorrt',
):  
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            print(f'{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument')
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(file, 'wb') as t:
        t.write(engine.serialize())
    return True