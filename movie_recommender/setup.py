"""Setup script for movie_recommender_3000"""

import os.path
from setuptools import setup

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name="movie_recommender_3000",
    version="0.9.0",
    description="Recommend movies based on sinmilarity & user taste",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/renegademonkey/misc_pers/movie_recommender",
    author="Rafael Wlodarski",
    author_email="renegademonkey@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    packages=["movie_recommender"],
    include_package_data=True,
    install_requires=[
        ""
    ],
    entry_points={"console_scripts": ["renegademonkey=movie_recommender.__main__:main"]},
)