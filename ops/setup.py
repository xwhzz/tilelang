from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='ops',
    ext_modules=[
        CppExtension(
            name='ops',
            sources=['kernels.cpp', 'vertical_slash_index.cu'],
            extra_compile_args=['-O3']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)