from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='docling-ollama-terminal',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A CLI tool for querying documents using Ollama and LlamaIndex',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/docling-ollama-terminal',
    packages=find_packages(),
    install_requires=[
        'llama-index-core==0.10.12',
        'llama-index-llms-ollama==0.1.2',
        'llama-index-embeddings-huggingface==0.1.2',
        'llama-index-readers-docling==0.1.0',
        'sentence-transformers==2.3.1',
        'transformers==4.37.2',
        'torch==2.1.2',
        'typing-extensions==4.9.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'docling-query=cli:main',
        ],
    },
)
