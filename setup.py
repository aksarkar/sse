import setuptools

setuptools.setup(
    name='sse',
    description='Sum of Single Effects Regression',
    version='0.1',
    url='https://github.com/aksarkar/sse',
    author='Abhishek Sarkar',
    author_email='aksarkar@uchicago.edu',
    license='MIT',
    install_requires=['matplotlib', 'numpy', 'pandas', 'scipy'],
    entry_points={
        'console_scripts': [
        ]
    }
)
