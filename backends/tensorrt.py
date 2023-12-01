
import numpy as np
import torch
import os
import numpy as np
import torch

from collections import namedtuple, OrderedDict


Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

class HostDeviceMem:
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TensorRTBackend:
    def __init__(self, half: bool = False):
        import tensorrt as trt
        # Enable tensorrt lazy loading
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.engine         = None
        self.context        = None
        self.bindings       = None
        self.binding_addrs  = None
        self.output_names   = None
        self.fp16           = None
        self.dynamic        = None
        self.batch_size     = None
        self.half = half

    
    def load(self, model_path: str, **kargs):
        import tensorrt as trt
        device = torch.device('cuda:0')
        
        with open(model_path, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            if engine.binding_is_input(i):
                if -1 in tuple(engine.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(engine.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items()) 
        # if dynamic, this is instead max batch size
        batch_size = bindings['images'].shape[0]

        self.engine         = engine
        self.context        = context
        self.bindings       = bindings
        self.binding_addrs  = binding_addrs
        self.output_names   = output_names
        self.fp16           = fp16
        self.dynamic        = dynamic
        self.batch_size     = batch_size
        return self
    

    def __call__(self, im):
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.engine.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images'] \
                ._replace(shape=im.shape)
            
            for name in self.output_names:
                i = self.engine.get_binding_index(name)
                self.bindings[name].data.resize_(
                    tuple(self.context.get_binding_shape(i))
                )
        
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' \
            if self.dynamic else 'not equal to'} max model size {s}"
        
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        
        return [self.bindings[x].data for x in sorted(self.output_names)]


class OldTensorRT:
    def __init__(self, half: bool = False):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.input_name = "input_0"
        self.output_name = "output_0"
        
        self.engine         = None
        self.context        = None
        self.bindings       = None
        self.binding_addrs  = None
        self.output_names   = None
        self.fp16           = None
        self.dynamic        = None
        self.batch_size     = None
        self.half = half
        # self.input_shape = (1, 3, 224, 224)
        # self.output_shape = (1, 17, 3)

    def load(self, model_path: str, **kargs):
        import tensorrt as trt
        import pycuda.driver as cuda    
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.stream()
        # Current NMS implementation in TRT only supports DataType.FLOAT but
        # it may change in the future, which could brake this sample here
        # when using lower precision [e.g. NMS output would not be np.float32
        # anymore, even though this is assumed in binding_to_type
        
        binding_to_type = {
            self.input_name: np.float32,
            self.output_name: np.float32
        }

        with open(model_path, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        
        for binding in engine:
            size = trt.volume(
                engine.get_binding_shape(binding)
            ) * engine.max_batch_size

            dtype = binding_to_type[str(binding)]
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device self.bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

        self.context = engine.create_execution_context()
    
    def inference(self, im: np.ndarray):
        import pycuda.driver as cuda 
        
        batch_size = im.shape[0]
        np.copyto(self.inputs[0].host, self.im.ravel())
        
        # Transfer input data to the GPU.
        for input in self.inputs:
            cuda.memcpy_htod_async(input.device, input.host, self.stream)
        # Run inference.
        self.context.execute_async(
            batch_size=batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        # Transfer predictions back from the GPU.
        for ouput in self.outputs:
            cuda.memcpy_dtoh_async(ouput.host, ouput.device, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        ouput = [out.host for out in self.outputs]# [:1]
        results = [
            heatmaps.reshape(1, -1, im.shape[2] // 4, im.shape[3] // 4) 
            for heatmaps in ouput
        ]
        return results[0] if len(results) == 1 else results