import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def jsonCopy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def readJsonObject(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f'{path.name} exceeds the permitted metadata size')
    value = jsonCopy(json.loads(path.read_text(encoding='utf-8')))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value


def validateCommittedTask(manifest, directory):
    terminal = readJsonObject(Path(directory) / 'result.json')
    if (terminal.get('state') != 'succeeded' or type(terminal.get('apiVersion')) is not int or
            terminal['apiVersion'] != 1 or
            any(not isinstance(manifest.get(key), str) or not manifest[key] or
                terminal.get(key) != manifest[key] for key in ('taskId', 'nodeId'))):
        raise ValueError('model does not belong to a successfully committed task')
    reference = terminal.get('modelReference')
    if not isinstance(reference, str) or reference.replace('\\', '/').split('/')[-1] != 'manifest.json':
        raise ValueError('task result does not reference a committed model manifest')


def atomicJson(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def fileDigest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _HashingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        written = self.stream.write(data)
        self.digest.update(memoryview(data).cast('B')[:written])
        return written

    def tell(self):
        return self.stream.tell()

    def flush(self):
        self.stream.flush()


def atomicArray(path, array):
    import numpy as np

    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as stream:
            writer = _HashingWriter(stream)
            if array.dtype.hasobject:
                raise ValueError('Object arrays cannot be saved when allow_pickle=False')
            if array.dtype.fields is not None:
                np.save(writer, array, allow_pickle=False)
            else:
                header = np.lib.format.header_data_from_array_1_0(array)
                try:
                    np.lib.format.write_array_header_1_0(writer, header)
                except ValueError:
                    np.lib.format.write_array_header_2_0(writer, header)
                for chunk in np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                                       op_flags=['readonly'], order='F' if header['fortran_order'] else 'C',
                                       buffersize=max(1, 1024 ** 2 // max(array.dtype.itemsize, 1))):
                    writer.write(chunk.tobytes(order='C'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {
        'file': path.name,
        'sha256': writer.digest.hexdigest(),
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }
