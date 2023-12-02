import onnxruntime as ort
from .backend import Backend


class ONNXBackend(Backend):
    def __init__(self) -> None:
        super().__init__()
        self.providers = ort.get_available_providers()
        self.session = None
        self.input_name = None
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
    def load(self, path: str, **kargs):
        super().load(path, **kargs)
        assert path.endswith('.onnx'), \
            f"File {path} is not a .onnx file"
        self.session = ort.InferenceSession(path, ort.InferenceSession(
                path,
                providers=self.providers
        ))
        self.input_name = self.session.get_inputs()[0].name

        return self
    
    def __call__(self, im):
        return self.session.run(None, {self.input_name: im})[0]

    