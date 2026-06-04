"""Setup script for Pcap2Rule."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pcap2rule",
    version="1.0.0",
    author="Pcap2Rule Team",
    description="Multi-Modal LLM Agent for Automated Suricata Rule Generation from PCAP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['pcap2rule'] + ['pcap2rule.' + p for p in find_packages()],
    package_dir={'pcap2rule': '.'},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "scapy>=2.5.0",
        "nfstream>=6.5.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "faiss-cpu>=1.7.4",
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.7.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "pcap2rule=pcap2rule.main:main",
        ],
    },
)
