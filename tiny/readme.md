
Please make sure you have TensorRT and pycuda installed.
For installation instructions, see:
- https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
- https://developer.nvidia.com/pycuda"""



from setuptools import setup, find_packages

setup(
    name='easy_ViTPose',
    author="JunkyByte",
    author_email="adriano.donninelli@hotmail.it",
    version='1.0',
    url="https://github.com/JunkyByte/easy_ViTPose",
    packages=find_packages(include=['easy_ViTPose', 'easy_ViTPose.*']),
)
