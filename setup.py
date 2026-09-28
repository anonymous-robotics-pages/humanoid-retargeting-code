from setuptools import setup, find_packages

setup(
  name = 'collision_free_motion_retargeting',
  packages = find_packages(),
  author="Anonymous Authors",
  description="Collision-Free Motion Retargeting for Humanoid Robots",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  license="MIT",
  version="0.1.0",
  install_requires=[
    "loop_rate_limiters",
    "mink",
    "mujoco",
    "numpy",
    "scipy",
    "pyyaml",
    "qpsolvers",
    "daqp",
    "torch",
    "rich",
    "tqdm",
    "pillow",
    "smplx @ git+https://github.com/vchoutas/smplx",
    "imageio[ffmpeg]",
  ],
  python_requires='>=3.10',
)
