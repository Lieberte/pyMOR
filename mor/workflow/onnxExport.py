from pathlib import Path

import numpy as np

from .storage import atomicArray, fileDigest


def exportInference(manifest, directory):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    directory = Path(directory)
    arrays = {key: np.load(directory / ref['file'], allow_pickle=False)
              for key, ref in manifest['arrays'].items()}
    kind = manifest['kind']
    nodes, initializers = [], []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(value), name))
        return name

    def operation(op, inputs, output, **attributes):
        nodes.append(helper.make_node(op, inputs, [output], **attributes))
        return output

    def linear(source, weight, bias, name):
        product = operation('MatMul', [constant(name + 'Weight', weight), source], name + 'Product')
        return operation('Add', [product, constant(name + 'Bias', bias[:, None])], name)

    extraInputs = []
    metadata = {'schemaVersion': 1, 'runtime': 'onnxruntime', 'dtype': 'float64',
                'axes': ['variable', 'sample'], 'inputName': 'values', 'outputName': 'prediction'}
    if kind == 'pod':
        n = arrays['basis'].shape[0]
        centered = operation('Sub', ['values', constant('mean', arrays['mean'][:, None])], 'centered')
        coefficients = operation('MatMul', [constant('encoder', arrays['basis'].T), centered], 'coefficients')
        decoded = operation('MatMul', [constant('decoder', arrays['basis']), coefficients], 'decoded')
        operation('Add', [decoded, 'mean'], 'prediction')
        outputRows = n
        metadata['mode'] = 'reconstruction'
    elif kind == 'autoencoder':
        n = manifest['architecture']['stateDim']
        current = operation('Cast', ['values'], 'floatValues', to=TensorProto.FLOAT)
        layers = [key[:-7] for key in arrays if key.endswith('.weight')]
        for index, key in enumerate(layers):
            current = linear(current, arrays[key + '.weight'], arrays[key + '.bias'], f'layer{index}')
            if index in (0, 2):
                current = operation('Relu', [current], f'relu{index}')
        operation('Cast', [current], 'prediction', to=TensorProto.DOUBLE)
        outputRows = n
        metadata['mode'] = 'reconstruction'
    elif kind in ('dmd', 'pod-dmd'):
        modes, eigenvalues = arrays['modes'], arrays['eigenvalues']
        inverse = np.linalg.pinv(modes)
        evolved = modes * eigenvalues[None, :]
        # Reject an unpaired complex truncation: silently dropping its imaginary
        # response changes the trained model. Check in bounded column blocks.
        for start in range(0, inverse.shape[1], 128):
            block = evolved @ inverse[:, start:start + 128]
            if np.linalg.norm(block.imag) > 1e-8 * max(np.linalg.norm(block), 1e-30):
                raise ValueError('DMD rank cuts a complex conjugate pair; choose a rank retaining complete pairs for real ONNX inference')
        n = modes.shape[0]
        encoder = np.vstack((inverse.real, inverse.imag))
        decoder = np.hstack((evolved.real, -evolved.imag))
        centered = operation('Sub', ['values', constant('mean', arrays['mean'][:, None])], 'centered')
        coefficients = operation('MatMul', [constant('encoder', encoder), centered], 'coefficients')
        decoded = operation('MatMul', [constant('decoder', decoder), coefficients], 'decoded')
        operation('Add', [decoded, 'mean'], 'prediction')
        outputRows = n
        metadata.update(mode='oneStep', dt=manifest['dt'], complexRepresentation='paired-real-factors')
    elif kind == 'bt':
        rank, inputs = arrays['B'].shape
        n = rank + inputs
        outputRows = rank + arrays['C'].shape[0]
        generator = np.hstack((np.linalg.solve(arrays['E'], arrays['A']),
                               np.linalg.solve(arrays['E'], arrays['B'])))
        metadata['systemMatrix'] = atomicArray(directory / 'inference-system.npy', generator)
        metadata.update(mode='response', stateCount=rank, inputCount=inputs,
                        outputCount=arrays['C'].shape[0], isContinuous=manifest['isContinuous'],
                        dt=manifest.get('dt'), initialState='zero', discretization='zero-order-hold')
        extraInputs.append(helper.make_tensor_value_info('transition', TensorProto.DOUBLE, [rank, n]))
        operation('MatMul', ['transition', 'values'], 'nextState')
        operation('MatMul', [constant('outputMatrix', np.hstack((arrays['C'], arrays['D']))), 'values'], 'output')
        operation('Concat', ['nextState', 'output'], 'prediction', axis=0)
    else:
        raise ValueError(f'No ONNX inference adapter for {kind}')
    graph = helper.make_graph(nodes, 'shonTA-ROM',
        [helper.make_tensor_value_info('values', TensorProto.DOUBLE, [n, 'samples'])] + extraInputs,
        [helper.make_tensor_value_info('prediction', TensorProto.DOUBLE, [outputRows, 'samples'])], initializers)
    model = helper.make_model(graph, producer_name='mor', opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    path = directory / 'inference.onnx'
    onnx.save_model(model, path)
    metadata.update(file=path.name, sha256=fileDigest(path), inputRows=n, outputRows=outputRows)
    return metadata
