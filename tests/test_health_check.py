
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from health_check import check_collection, check_qdrant_connection


def test_qdrant_connection_success(capsys):
    client = Mock()
    client.get_collections.return_value = []

    result = check_qdrant_connection(client)

    assert result is True
    assert "[OK] Qdrant is reachable." in capsys.readouterr().out


def test_qdrant_connection_failure(capsys):
    client = Mock()
    client.get_collections.side_effect = ConnectionError("Unavailable")

    result = check_qdrant_connection(client)

    output = capsys.readouterr().out

    assert result is False
    assert "[ERROR] Cannot connect to Qdrant." in output


def test_collection_with_points(capsys):
    client = Mock()
    client.get_collection.return_value = SimpleNamespace(
        points_count=310
    )

    result = check_collection(client)

    output = capsys.readouterr().out

    assert result is True
    assert "Stored points: 310" in output


def test_empty_collection(capsys):
    client = Mock()
    client.get_collection.return_value = SimpleNamespace(
        points_count=0
    )

    result = check_collection(client)

    output = capsys.readouterr().out

    assert result is False
    assert "The collection is empty." in output
