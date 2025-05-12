# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

import torch
from ..base import BaseKernelAdapter
import ctypes
from typing import List, Optional, Union, Callable, Dict, Tuple, Any
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.relay import TensorType
from tvm import tir
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module


class CtypesKernelAdapter(BaseKernelAdapter):
    """Adapter class that converts TVM/TIR functions to callable CUDA kernels using ctypes.
    
    This adapter handles:
    1. Converting TIR functions to compiled CUDA libraries
    2. Managing dynamic shapes in tensor operations
    3. Wrapping C++ kernels for Python/PyTorch usage
    """

    # Class attributes to store compiled kernel information
    target = "cuda"
    ir_module: Optional[tvm.IRModule] = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    kernel_global_source: Optional[str] = None
    lib: Optional[ctypes.CDLL] = None  # Compiled library handle
    wrapped_source: Optional[str] = None  # Generated C++ wrapper code
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: Optional[Dict[tir.Var, Tuple[int, int]]] = None
    # Pass configs for the compiler
    pass_configs: Optional[Dict[str, Any]] = None

    # Add new cache attributes
    param_dtypes: Optional[List[torch.dtype]] = None  # Cache for parameter dtypes
    param_shapes: Optional[List[List]] = None  # Cache for parameter shapes

    def __init__(self,
                 params: List[TensorType],
                 result_idx: List[int],
                 target: str,
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 host_mod: Optional[tvm.IRModule] = None,
                 device_mod: Optional[tvm.IRModule] = None,
                 kernel_global_source: Optional[str] = None,
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None):
        """Initialize the adapter with the given TIR function or module.
        
        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        # Cache parameter information during initialization
        self.param_dtypes = [param.dtype for param in params]
        self.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            self.param_shapes.append(native_shape)

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()

        self.target = Target.canon_target(determine_target(target))
        self.verbose = verbose
        self.wrapper = TLWrapper(self.target)
        self.lib_generator = LibraryGenerator(self.target)

        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))

        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()
        self.lib.init()

        self._post_init()

    @classmethod
    def from_database(cls,
                      params: List[TensorType],
                      result_idx: List[int],
                      target: str,
                      func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                      kernel_global_source: str,
                      kernel_lib_path: str,
                      verbose: bool = False,
                      pass_configs: Optional[Dict[str, Any]] = None):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.kernel_global_source = kernel_global_source
        adapter.wrapped_source = kernel_global_source
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tir.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        # Cache parameter information during initialization
        adapter.param_dtypes = [param.dtype for param in params]
        adapter.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            adapter.param_shapes.append(native_shape)

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()

        adapter.target = Target.canon_target(determine_target(target))
        adapter.verbose = verbose
        adapter.lib_generator = LibraryGenerator(adapter.target)
        adapter.lib = adapter.lib_generator.load_lib(lib_path=kernel_lib_path)
        adapter.lib.init()

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self):
        """Extract information about dynamic shapes from the TIR function.
        
        Maps symbolic variables to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                    dynamic_symbolic_map[shape] = (i, j)
        return dynamic_symbolic_map

    def _forward_from_prebuild_lib(self, *args, stream: Optional[int] = None):
        """Low-level function to call the compiled CUDA kernel.
        
        Converts PyTorch tensor pointers to C void pointers for ctypes interface.
        """
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def _warp_forward_from_prebuild_lib(self,
                                        *ins: List[torch.Tensor],
                                        stream: Optional[int] = None):
        """High-level wrapper for kernel execution.
        
        Handles:
        1. Input validation
        2. Output tensor allocation
        3. Dynamic shape resolution
        4. CUDA stream management
        
        Args:
            ins: Input PyTorch tensors
            stream: Optional CUDA stream for asynchronous execution
        
        Returns:
            Single tensor or list of tensors containing the kernel results
        """
        if len(ins) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
            )
        ins_idx = 0
        args = []

        # tensor pointers
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for s in self.param_shapes[i]:
                    if isinstance(s, tir.Var):
                        ref_tensor_idx, ref_shape_idx = self.dynamic_symbolic_map[s]
                        shape.append(ins[ref_tensor_idx].shape[ref_shape_idx])
                    else:  # Already converted to Python int during initialization
                        shape.append(s)
                device = ins[0].device if len(ins) > 0 else torch.cuda.current_device()
                tensor = torch.empty(*shape, dtype=dtype, device=device)
            else:
                tensor = ins[ins_idx]
                ins_idx += 1
            args.append(tensor)

        # dynamic symbolics
        for _, (buffer_idx, shape_idx) in self.dynamic_symbolic_map.items():
            args.append(ins[buffer_idx].shape[shape_idx])

        # if stream is not None, we need to pass the stream to the library
        if stream is None:
            if str(self.target).startswith("cuda") and torch.cuda.is_available():
                stream = torch.cuda.current_stream().cuda_stream
            else:
                stream = 0

        self._forward_from_prebuild_lib(*args, stream=stream)

        if len(self.result_idx) == 1:
            return args[self.result_idx[0]]
        else:
            return [args[i] for i in self.result_idx]

    def _convert_torch_func(self) -> Callable:
        """Returns a PyTorch-compatible function wrapper for the kernel."""
        return self._warp_forward_from_prebuild_lib

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)

    @property
    def srcpath(self):
        """Returns the source path of the compiled library."""
        return self.lib_generator.srcpath

    @property
    def libpath(self):
        """Returns the path to the compiled library."""
        return self.lib_generator.libpath

    @property
    def lib_code(self):
        """Returns the code of the compiled library."""
        return self.lib_generator.lib_code

    @property
    def is_dynamic(self):
        """Indicates whether the kernel handles dynamic shapes."""
        return (self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0)

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.kernel_global_source
        else:
            assert self.wrapped_source is not None, "Wrapped source is not available"
            return self.wrapped_source
