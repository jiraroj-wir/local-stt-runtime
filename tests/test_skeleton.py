from app import PROJECT_NAME


def test_project_name_constant() -> None:
    assert PROJECT_NAME == "local-stt-runtime"
