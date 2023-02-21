from setuptools import setup

setup(
    name="AceCE",
    version="1.0.0",
    packages=[
        "AceCE",
        "AceCE.utils",
        "AceCE.sample_generators",
        "AceCE.specie_decorators",
    ],
    url="https://cedergrouphub.github.io/AceCE",
    license="Modified BSD",
    author="Fengyu Xie",
    author_email="fengyu_xie@berkeley.edu",
    description="An automated workflow implementation of cluster expansion"
    " in crystalline solids based on Atomate2, "
    "Jobflow and smol. ",
)
