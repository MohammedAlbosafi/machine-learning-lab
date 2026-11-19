from importlib.metadata import version


def test_project_package_is_installed():
    import digit_recognition

    assert digit_recognition.__file__ is not None
    assert version("digit-recognition")
