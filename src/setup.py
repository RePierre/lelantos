from setuptools import setup, find_packages

version = '0.1'

install_requires = [
    'numpy',
    'keras'
]

dev_requires = [
    'python-language-server[all]'
]

tests_requires = [
]

setup(
    name='lelantos',
    version=version,
    description="A deep learning attempt at splitting phrases into clauses.",
    long_description="",
    classifiers=[],
    keywords="",
    author="RePierre",
    author_email="",
    url="",
    license="",
    packages=find_packages(exclude=['']),
    include_package_data=True,
    zip_safe=True,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require={
        'dev': dev_requires
    },
    test_suite="py.test",
)
