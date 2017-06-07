from deep_qa.common.checks import ensure_pythonhashseed_set

def test_pythonhashseed():
    ensure_pythonhashseed_set()
