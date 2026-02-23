import setuptools

setuptools.setup(
    name="ProfibusDp",
    version="0.1.0",
    author="Sebastian Jennen",
    author_email="sj@imagearts.de",
    description="Profibus DP device driver",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['requests'],
    scripts=['ProfibusDp.py']
)