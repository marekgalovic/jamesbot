from setuptools import setup, find_packages

setup(name='jamesbot',
  version='1',
  packages=find_packages(),
  description='JamesBot - v.1',
  author='Marek Galovic',
  author_email='galovic.galovic@gmail.com',
  license='MIT',
  install_requires=[
    'numpy',
    'nltk'
  ],
  zip_safe=False
)
