try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as f:
    readme = f.read()


setup(
    name='joint',
    version='0.0.1',
    description='JOINT for Large-scale Single-cell RNA-Sequencing Analysis via Soft-clustering and Parallel Computing.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Tao Cui',
    author_email='taocui.caltech@gmail.com',
    url='https://github.com/wanglab/joint',
    keywords='joint single cell clustering',
    packages=['joint'],
    include_package_data=True,
    install_requires=[
        'anndata',
        'scanpy',
        'pandas',
        'tensorflow',
        'numpy',
        'scipy',
        'scikit_learn',
    ],
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)