from setuptools import setup, find_packages

setup(
    name="livevoice",
    version="0.1.0",
    description="Streaming voice conversion via codec-based language modeling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "lightning>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "descript-audio-codec>=1.0.0",
        "transformers>=4.40.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "torchcrepe>=0.0.19",
        "pandas>=2.0.0",
        "tensorboard>=2.13.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "jiwer>=3.0.0",
        ],
    },
)
