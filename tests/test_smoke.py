import pytest

import guardlens
from guardlens.cli import PLANNED_COMMANDS, main
from guardlens.settings import Settings


def test_package_imports() -> None:
    assert guardlens.__version__


def test_settings_load_without_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_api_key is None
    assert settings.data_dir.name == "data"


def test_settings_read_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "placeholder-key"
    assert "placeholder-key" not in repr(settings)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("command", sorted(PLANNED_COMMANDS))
def test_planned_commands_report_not_implemented(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command]) == 2
    assert "not implemented yet" in capsys.readouterr().out
