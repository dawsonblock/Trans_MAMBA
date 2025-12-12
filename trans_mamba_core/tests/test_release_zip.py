import zipfile
from pathlib import Path


def test_release_zip_top_dir_and_excludes(tmp_path):
    from make_release_zip import DEFAULT_TOP_DIR, build_release_zip

    output_zip = tmp_path / f"{DEFAULT_TOP_DIR}.zip"

    build_release_zip(
        repo_root=Path(__file__).resolve().parents[2],
        output_zip=output_zip,
        top_dir=DEFAULT_TOP_DIR,
    )

    assert output_zip.exists()

    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()

    assert names

    top_prefix = f"{DEFAULT_TOP_DIR}/"
    assert all(n.startswith(top_prefix) for n in names)

    assert not any("/.git/" in n for n in names)
    assert not any("/legacy/" in n for n in names)
    assert not any("/__pycache__/" in n for n in names)
    assert not any(n.endswith(".pyc") for n in names)
