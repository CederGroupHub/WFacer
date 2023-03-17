from setuptools import setup

setup(
    name="WFacer",
    version="0.0.1",
    packages=[
        "WFacer",
        "WFacer.utils",
        "WFacer.sample_generators",
        "WFacer.specie_decorators",
    ],
    url="https://cedergrouphub.github.io/WFacer",
    license="Modified BSD",
    author="Fengyu Xie",
    author_email="fengyu_xie@berkeley.edu",
    description="An automated workflow implementation of cluster expansion"
    " in crystalline solids based on Atomate2, "
    "Jobflow and smol. ",
)
