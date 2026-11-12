from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docker_smoke.sh"


def test_smoke_does_not_pipe_curl_into_an_early_closing_head() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    unwrapped = script.replace("\\\n", " ")
    assert "curl" in unwrapped
    assert "| head -c" not in unwrapped
    for variable, limit in (
        ("health_response", 300),
        ("classify_response", 400),
        ("restored_response", 120),
    ):
        assert f'{variable}=$(curl -fsS' in unwrapped
        assert f"printf '%s\\n' \"${{{variable}:0:{limit}}}\"" in script
