# tests/test_env.py
def test_python():
    import sys
    assert sys.version_info.major >= 3

def test_imports():
    import numpy, pandas, matplotlib, sklearn
