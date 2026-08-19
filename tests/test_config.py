from src.config import env_flag


def test_env_flag_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_TEST_FLAG", raising=False)

    assert env_flag("RAG_TEST_FLAG", default=True) is True
    assert env_flag("RAG_TEST_FLAG", default=False) is False


def test_env_flag_falsy_values_override_default(monkeypatch):
    for value in ("0", "false", "FALSE", "no", "off", " off "):
        monkeypatch.setenv("RAG_TEST_FLAG", value)

        assert env_flag("RAG_TEST_FLAG", default=True) is False, value


def test_env_flag_truthy_values_override_default(monkeypatch):
    for value in ("1", "true", "yes", "on", "anything"):
        monkeypatch.setenv("RAG_TEST_FLAG", value)

        assert env_flag("RAG_TEST_FLAG", default=False) is True, value
