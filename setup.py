from setuptools import find_packages, setup

setup(
    name="your_module_name",
    version="0.0.1",
    packages=find_packages(exclude=[]),
    author="Kai-Hendrik Cohrs",
    author_email="kaicohrs@uv.es",
    url="https://github.com/KaiHCohrs/nn-repo-template",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English"
        "Topic :: Scientific/Engineering :: Machine Learning",
    ],
    # keywords="",
    # install_requires=requirements,
    # license="MIT",
    # entry_points={
    #    'console_scripts': [
    #        'new_repo=new_repo.main:main'
    #    ]
    # },
)
