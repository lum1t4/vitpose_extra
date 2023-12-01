from nn.model import ViTPose
import torch
import os
from .backend import Backend
COREML_MAX_BATCH_SIZE = 10


class CoreMLBackend(Backend):
    def __init__(self):
        super().__init__()
        try: 
            import coremltools as ct
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "CoreML is not supported on this device or "
                "you have not installed coremltools"
            )
        self.model = None
        self.metadata = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None

    def load(self, path: str, **kargs):
        super().load(path, **kargs)
        assert path.endswith('.mlmodel') or path.endswith('.mlpackage'), \
            f"File {path} is not a .mlmodel or .mlpackage file"
        import coremltools as ct
        self.model = ct.models.MLModel(path)
        self.metadata = dict(self.model.user_defined_metadata)
        self.input_name = self.model.get_spec().description.input[0].name
        self.output_name = self.model.get_spec().description.output[0].name
        self.input_shape = self.model.get_spec().description.input[0].type.multiArrayType.shape
        return self
    
    def __call__(self, inputs):
        return self.model.predict({self.input_name: inputs})[self.output_name]
