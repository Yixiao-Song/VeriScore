from setuptools import setup, find_packages

setup(
   name="VeriScore",
   version="2.0.2",
   packages=find_packages(),
   install_requires=[
      'spacy',
      'openai',
      'anthropic',
      'tiktoken',
      'tqdm'
   ],
   # author="Yixio Song",
   # author_email="yixiaosong@umass.edu",
   description="Pip package for VeriScore",
   long_description=open('README.md').read(),
   long_description_content_type="text/markdown",
   # url="https://github.com/Yixiao-Song/VeriScore",
   classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: MIT License",
       "Operating System :: OS Independent",
   ],
   python_requires='>=3.9',
)
