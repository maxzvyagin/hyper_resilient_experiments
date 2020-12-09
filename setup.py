"""Setup for the hyper_resilient_experiments package."""

import setuptools

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Max Zvyagin",
    author_email="max.zvyagin7@gmail.com",
    name='hyper_resilient_experiments',
    license="MIT",
    long_description_content_type="text/markdown",
    description='Experiments for hyper_resilient testing.',
    version='v0.0.1',
    long_description=README,
    url='https://github.com/maxzvyagin/hyper_resilient_experiments',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['scikit-learn==0.22.1', 'scikit-optimize==0.5.2', 'ray', "ray [tune]", "hyperspaces", "spaceray"],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)