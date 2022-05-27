
from setuptools import setup, find_packages
from nf4ip.core.version import get_version

VERSION = get_version()

f = open('README.md', 'r')
LONG_DESCRIPTION = f.read()
f.close()

setup(
    name='nf4ip',
    version=VERSION,
    description='NF4IP',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Nico Hoffmann',
    author_email='n.hoffmann@hzdr.de',
    url='https://github.com/Photon-AI-Research/NF4IP',
    license='BSD-3',
    packages=find_packages(exclude=['ez_setup', 'tests*']),
    package_data={'nf4ip': ['templates/*']},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        nf4ip = nf4ip.main:main
    """,
)
