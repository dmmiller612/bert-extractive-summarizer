from setuptools import setup
from setuptools import find_packages

setup(name='bert-extractive-summarizer',
      version='0.1.0',
      description='Extractive Text Summarization with BERT',
      keywords = ['bert', 'pytorch', 'machine learning', 'deep learning', 'extractive summarization'],
      url='https://github.com/dmmiller612/bert-extractive-summarizer',
      download_url='https://github.com/dmmiller612/bert-extractive-summarizer/archive/0.1.0.tar.gz',
      author='Derek Miller',
      author_email='dmmiller612@gmail.com',
      install_requires=['pytorch-pretrained-bert', 'sklearn', 'nltk'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
