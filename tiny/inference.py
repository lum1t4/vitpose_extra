import os
from enum import Enum
import numpy as np
import cv2
import torch

from easy_ViTPose.configs.ViTPose_common import data_cfg
from easy_ViTPose.vit_models.model import ViTPose
from easy_ViTPose.vit_utils.util import dyn_model_import
from easy_ViTPose.vit_utils.top_down_eval import keypoints_from_heatmaps
from easy_ViTPose.vit_utils.inference import pad_image

try:
    import tensorrt as trt
    import pycuda.driver as cuda
except ModuleNotFoundError as err:
    pass


class InferenceEnginge(Enum):
    pytorch = 0
    coreml = 1
    onnx = 2
    tensorrt = 3


class HostDeviceMem:
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()



class VitposeTensorRTInference:
    def __init__(self, engine_path: str):
        self.input_name = "input_0"
        # Name of output node
        self.output_name = "output_0"
        # CHW format of model input
        self.input_shape = (3, 256, 192)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine: trt.ICudaEngine = self.load_engine(engine_path)
        self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f:
            data = f.read()
        engine = self.runtime.deserialize_cuda_engine(data)
        return engine

    def save_engine(self, engine_path):
        print('Engine:', self.engine)
        buffer = self.engine.serialize()
        with open(engine_path, 'wb') as f:
            f.write(buffer)
    
    def allocate_buffers(self):
        """Allocates host and device buffer for TRT engine inference.

        This function is similair to the one in common.py, but
        converts network outputs (which are np.float32) appropriately
        before writing them to Python buffer. This is needed, since
        TensorRT plugins doesn't support output type description, and
        in our particular case, we use NMS plugin as network output.

        Args:
            engine (trt.ICudaEngine): TensorRT engine

        Returns:
            inputs [HostDeviceMem]: engine input memory
            outputs [HostDeviceMem]: engine output memory
            bindings [int]: buffer to device bindings
            stream (cuda.Stream): cuda stream for engine inference synchronization
        """
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Current NMS implementation in TRT only supports DataType.FLOAT but
        # it may change in the future, which could brake this sample here
        # when using lower precision [e.g. NMS output would not be np.float32
        # anymore, even though this is assumed in binding_to_type]
        binding_to_type = {self.input_name: np.float32, self.output_name: np.float32}

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = binding_to_type[str(binding)]
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
    
    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def _predict(self, x: np.ndarray, batch_size: int = 1):
        # Transfer input data to the GPU.
        for input in self.inputs:
            cuda.memcpy_htod_async(input.device, input.host, self.stream)
        # Run inference.
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        for ouput in self.outputs:
            cuda.memcpy_dtoh_async(ouput.host, ouput.device, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]
    
    def predict(self, img: np.ndarray):
        np.copyto(self.inputs[0].host, img.ravel())
        heatmaps = self._predict(img, batch_size=1)[0]
        # Reshape to output size
        heatmaps = heatmaps.reshape(1, -1, img.shape[2] // 4, img.shape[3] // 4)
        return heatmaps
    
    def build_engine(self, uff_model_path, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size: int = 1, silent=False):
        with trt.Builder(self.logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = 2 << 30
            builder.max_batch_size = batch_size
            if trt_engine_datatype == trt.DataType.HALF:
                builder.fp16_mode = True
            elif trt_engine_datatype == trt.DataType.INT8:
                raise NotImplementedError
                builder.fp16_mode = True
                builder.int8_mode = True
                builder.int8_calibrator = calibrator.SSDEntropyCalibrator(data_dir=calib_dataset, cache_file='INT8CacheFile')

            parser.register_input(self.input_name, self.input_shape)
            parser.register_output(self.output_name)
            parser.parse(uff_model_path, network)

            if not silent:
                print("Building TensorRT engine. This may take few minutes.")

            return builder.build_cuda_engine(network)


class VitPoseInference:
    def __init__(self, model_path: str, engine: InferenceEnginge = InferenceEnginge.pytorch,
                 model_name: str = 's', model_dataset: str = 'coco_25'):
        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'
        
        self.model_name = model_name
        self.model_dataset = model_dataset
        self.engine_type = engine
        self.model_path = model_path
        self.img_size = data_cfg['image_size']
        self._load_model()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_model(self):
        match self.engine_type:
            case InferenceEnginge.coreml:
                import coremltools as ct
                self.model = ct.models.MLModel(self.model_path)
                self.predict = lambda x: self.model.predict({'x_1': x})['var_748']
            case InferenceEnginge.pytorch:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                from easy_ViTPose.vit_models.model import ViTPose
                model_cfg = dyn_model_import(self.model_dataset, self.model_name)
                self.model = ViTPose(model_cfg)
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
                self.model.to(self.device)
                self.model.eval()
                self.predict = lambda x: self.model(torch.from_numpy(x).to(self.device)).to('cpu').detach().numpy()
            case InferenceEnginge.onnx:
                raise NotImplementedError("ONNX inference not implemented yet")
            case InferenceEnginge.tensorrt:
                self.model = VitposeTensorRTInference(self.model_path)
                self.predict = lambda x: self.model.predict(x)
                pass

    def inference(self, img: np.ndarray, bbox: np.ndarray):
        cropped_img, org_h, org_w, left_pad, top_pad = self.prepocess(img, bbox)
        heatmaps = self.predict(cropped_img)
        keypoints = np.squeeze(self.postprocess(heatmaps, org_w, org_h))
        keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
        return keypoints

    def prepocess(self, img: np.ndarray, bbox: np.ndarray):
        bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-10, 10], 0, img.shape[1])
        bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-10, 10], 0, img.shape[0])
        # Crop image and pad to 3/4 aspect ratio
        img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img, (left_pad, top_pad) = pad_image(img, 3 / 4)
        org_h, org_w = img.shape[:2]
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR) / 255
        img = np.expand_dims(((img - self.mean) / self.std).transpose(2, 0, 1), axis=0).astype(np.float32)
        return img, org_h, org_w, left_pad, top_pad
    
    def postprocess(self, heatmaps, org_w, org_h):
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[org_w // 2, org_h // 2]]),
            scale=np.array([[org_w, org_h]]),
            unbiased=True,
            use_udp=True
        )
        return np.concatenate([points[:, :, ::-1], prob], axis=2)