from setuptools import setup

setup(
    name='mpc_trajopt',
    version='0.3.0',
    packages=['mpc_trajopt'],
    license='MIT',
    author='Alexander (Sasha) Lambert',
    author_email='lambert.sasha@gmail.com',
    description='Parallel trajectory optimization and MPC algorithms in pytorch.',
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'gym',
        'tensorboard',
    ],
)

