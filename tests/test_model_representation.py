"""fetch_model_representation uses type=proc, not raw SQL."""

from inference.models import MODEL_REPRESENTATION_PROC, fetch_model_representation


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def proc(self, name: str, value: object) -> list[dict[str, str]]:
        self.calls.append((name, value))
        return [{"model": str((value or {}).get("models", ["?"])[0])}]


def test_fetch_model_representation_calls_proc_with_models() -> None:
    client = FakeClient()
    rows = fetch_model_representation(client, ["40724_R", "  ", ""])
    assert rows == [{"model": "40724_R"}]
    assert client.calls == [(MODEL_REPRESENTATION_PROC, {"models": ["40724_R"]})]


def test_fetch_model_representation_empty_codes_skips_proc() -> None:
    client = FakeClient()
    assert fetch_model_representation(client, ["", "  "]) == []
    assert client.calls == []
