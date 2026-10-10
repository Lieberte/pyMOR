import hashlib
import io

import numpy as np
import pytest

from mor.workflow import storage


@pytest.mark.parametrize('array', [
    np.arange(15., dtype=np.float64).reshape(3, 5),
    np.asfortranarray(np.arange(15.).reshape(3, 5)),
    np.arange(40).reshape(5, 8)[::2, ::2],
    np.arange(10)[::-1], np.array(2.5), np.empty((0, 4)),
    np.array([1 + 2j, 3 - 1j]), np.arange(9, dtype='>f8'),
    np.array([(1., 2)], dtype=[('温度', 'f8'), ('id', 'i4')]),
])
def test_streamed_array_has_identical_npy_bytes_and_digest(tmp_path, array, monkeypatch):
    expected = io.BytesIO()
    np.save(expected, array, allow_pickle=False)
    monkeypatch.setattr(storage, 'fileDigest', lambda path: pytest.fail('reread saved array'))
    path = tmp_path / 'array.npy'
    metadata = storage.atomicArray(path, array)
    assert path.read_bytes() == expected.getvalue()
    assert metadata['sha256'] == hashlib.sha256(expected.getvalue()).hexdigest()
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), array)


def test_array_write_failure_keeps_previous_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / 'array.npy'
    path.write_bytes(b'previous')

    def failReplace(*args):
        raise OSError('simulated replacement failure')

    monkeypatch.setattr(storage.os, 'replace', failReplace)
    with pytest.raises(OSError, match='replacement failure'):
        storage.atomicArray(path, np.ones((4, 3)))
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [path]
