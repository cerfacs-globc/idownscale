from pathlib import Path

from iriscc.provenance import describe_path, inventory_paths


def test_describe_path_reports_existing_file_metadata(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("abc")
    info = describe_path(path)
    assert info["path"] == str(path)
    assert info["exists"] is True
    assert info["is_file"] is True
    assert info["size_bytes"] == 3
    assert "mtime" in info


def test_inventory_paths_supports_single_paths_and_lists(tmp_path):
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("l")
    right.write_text("rr")
    inventory = inventory_paths({"single": left, "many": [left, right]})
    assert inventory["single"]["path"] == str(left)
    assert inventory["many"][0]["path"] == str(left)
    assert inventory["many"][1]["size_bytes"] == 2
