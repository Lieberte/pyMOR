"""Restore constant linear thermal physics from a STA archive and exported mesh.

Result temperatures and solved boundary heat flows are deliberately never read.
Network surfaces use area-averaged FEM temperatures and uniform distributed
loads; their zero heat capacity is eliminated rather than approximated by a mass.
"""
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import csr_matrix

from mor.operators.lowRank import lowRankOperator
from mor.operators.matrix import matrixOperator
from mor.operators.sum import sumOperator
from mor.operators.thermalSet import thermalOperatorSet
from .coupling import thermalCouplingInterface
from .surfaceMapping import closestTriangleTrace
from .thermal import assembleLinearTetraThermalOperators


def _fixed(table, label):
    if table.get('type') != 'fixedValue' or 'fixedValue' not in table:
        raise NotImplementedError(f'{label} requires an explicit constant fixedValue')
    value = float(table['fixedValue'])
    if not np.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return value


def _property(control, material, name):
    path = f'{material}/{name}/isotropic/fixedValue'
    values = np.asarray(control[path], float).ravel()
    if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
        raise ValueError(f'{path} must contain one positive material property')
    return float(values[0])


def _operatorSet(k, m, q, volumes=()):
    return thermalOperatorSet(k, matrixOperator(m, 'scipy'), matrixOperator(q, 'scipy'), np.asarray(volumes))


def _spatialConvectionTrace(patch, nodes, coefficients):
    """Three-point face quadrature exactly integrates constant h times Ni*Nj."""
    xyz = nodes[patch.triangles]
    areas = .5 * np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)
    coefficients = np.asarray(coefficients, float).ravel()
    if coefficients.shape != areas.shape or not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0):
        raise ValueError('mapped spatial HTC must contain one finite nonnegative value per patch triangle')
    if np.any(areas <= 0):
        raise ValueError('spatial convection contains degenerate triangles')
    shape = np.full((3, 3), 1/6)
    np.fill_diagonal(shape, 2/3)
    columns = np.repeat(patch.triangles, 3, axis=0)
    weights = np.tile(shape, (len(areas), 1))
    rows = np.repeat(np.arange(3*len(areas)), 3)
    trace = csr_matrix((weights.ravel(), (rows, columns.ravel())), shape=(3*len(areas), patch.order))
    conductance = np.repeat(areas * coefficients / 3, 3)
    return trace, conductance


@dataclass
class thermalPatch:
    nodes: np.ndarray
    areas: np.ndarray
    triangles: np.ndarray
    order: int

    @property
    def area(self):
        return float(self.areas.sum())

    def averageTrace(self):
        return csr_matrix((self.areas / self.area, (np.zeros(len(self.nodes), int), self.nodes)),
                          shape=(1, self.order))


@dataclass
class thermalArchiveAssembly:
    components: dict = field(default_factory=dict)
    interfaces: list = field(default_factory=list)
    initialTemperatures: dict = field(default_factory=dict)
    nodes: dict = field(default_factory=dict)
    patches: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)

    def reduce(self, bases, *, rank=None):
        """Project restored physics; the caller supplies offline bases only."""
        from mor.reductors.thermalCoupling import thermalCouplingReductor
        reductor = thermalCouplingReductor()
        if set(bases) != set(self.components):
            raise ValueError('bases must specify every physical component exactly once')
        components = [reductor.reduceComponent(name, model, basis=bases[name], rank=rank)
                      for name, model in self.components.items()]
        return reductor.assemble(components, self.interfaces)


def loadThermalArchive(staPath, componentFolders, *, capacityMode='consistent', progress=None,
                       useSpatialConvection=True):
    """Assemble active direct-schema FEM solids, contacts and a constant network.

    ``componentFolders`` maps caller-selected names to exported full-mesh cases.
    Unknown physics fails explicitly. Search radii are taken from the archive;
    uncovered contact quadrature area is reported rather than bridged silently.
    """
    import h5py
    from mor.models.thermal import thermalModel
    result = thermalArchiveAssembly(audit={'sourceSta': str(staPath), 'components': [],
                                         'contacts': [], 'network': [], 'prescribedLoads': [],
                                         'resultFieldsUsedAsInputs': False,
                                         'networkTrace': 'areaAverage', 'capacityMode': capacityMode,
                                         'useSpatialConvection': useSpatialConvection})
    with zipfile.ZipFile(staPath) as archive:
        control = json.loads(archive.read('controlData.json'))
        system = control['rootSystem']
        if system.get('children'):
            raise NotImplementedError('nested systems require explicit thermal network flattening')
        thermal = system['thermalComponents']
        supported = {'femRegions', 'femFemTccInterfaces', 'femNetworkInterfaces', 'thermalMasses',
                     'thermalSurfaces', 'thermalSources', 'convections', 'conductions'}
        for category, records in thermal.items():
            if category not in supported and any(v.get('isActive', True) for v in records.values()):
                raise NotImplementedError(f'unsupported active thermal category: {category}')
        regionNames = {}
        spatialConvections = {}
        with h5py.File(io.BytesIO(archive.read('controlData.h5')), 'r') as properties:
            for name, folder in componentFolders.items():
                if progress:
                    progress(f'assembling {name}')
                folder = Path(folder)
                manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8-sig'))
                uid = manifest['regionUuid']
                region = thermal['femRegions'][uid]
                if not region.get('isActive', True) or uid in regionNames:
                    raise ValueError('exported components must map one-to-one to active regions')
                regionNames[uid] = name
                with np.load(folder / manifest['dataFile'], allow_pickle=False) as data:
                    nodes = data['nodesMeters'].copy()
                    tetra = data['tetrahedra'].copy()
                    surface = data['surfaceTriangles'].copy()
                    volumes = data['elementVolumesM3'].copy()
                    exportedConductivity = data['conductivityTensorWPerMK'].copy()
                    sampleNodes = data['boundarySampleNodeIndices'].copy()
                    sampleAreas = data['boundaryWeightsM2'].copy()
                    samplePatches = data['boundaryPatchIds'].copy()
                material = region['materialUuid']
                rho, cp = (_property(properties, material, prop) for prop in ('density', 'specificHeatCapacity'))
                propertyDefinition = control['material']['solids'][material]['thermalConductivity']
                if (propertyDefinition['type'] != 'isotropic'
                        or propertyDefinition['isotropic']['type'] != 'fixedValue'):
                    raise NotImplementedError('archive adapter currently supports constant isotropic solid conductivity')
                conductivity = np.eye(3) * _property(properties, material, 'thermalConductivity')
                if not np.allclose(exportedConductivity, conductivity, rtol=1e-10, atol=1e-12):
                    raise ValueError('exported conductivity differs from source archive')
                source = np.zeros(len(tetra))
                with h5py.File(io.BytesIO(archive.read(f"mesh/{region['meshUuid']}.samesh")), 'r') as mesh:
                    if (not np.array_equal(np.asarray(mesh['nodes']).reshape(-1, 3), nodes)
                            or not np.array_equal(np.asarray(mesh['connectivityVolume']).reshape(-1, 4), tetra)
                            or not np.array_equal(np.asarray(mesh['connectivitySurface']).reshape(-1, 3), surface)):
                        raise ValueError(f'{name}: exported connectivity differs from source archive')
                    # Read source loads rather than the exported solved heat-flow fields.
                    for patch in region.get('volumePatches', {}).values():
                        for child, group in patch['children'].items():
                            heat = group.get('volumetricGroups', {}).get('heatSource')
                            if heat is None:
                                continue
                            if heat.get('mode', 'table1D') != 'table1D':
                                raise NotImplementedError('spatial volume sources are not supported')
                            value = _fixed(heat['heatSource1D'], f'{name} volume source')
                            elements = np.asarray(mesh[f'volumes/{child}'], int).ravel()
                            if not len(elements) or np.any(elements < 0) or np.any(elements >= len(tetra)):
                                raise ValueError('invalid source volume element indices')
                            variable = heat['inputVariable']
                            if variable == 'power':
                                source[elements] += value / volumes[elements].sum()
                            elif variable in ('volumetricHeatSource', 'heatSource'):
                                source[elements] += value
                            else:
                                raise NotImplementedError(f'unsupported volume source: {variable}')
                    for patch in manifest['metadata']['surfacePatches']:
                        patchUid = patch['patchUuid']
                        raw = region['patches'][patch['parentPatchUuid']]['children'][patchUid]
                        boundary = raw.get('boundaryConditions', {})
                        if boundary.get('constT') or boundary.get('convective'):
                            raise NotImplementedError('direct temperature/convection boundaries require an explicit adapter')
                        heat = boundary.get('heatFlux')
                        if heat and (heat['mode'] != 'table1D' or _fixed(heat['heatFlux1D'], 'direct boundary flux') != 0):
                            raise NotImplementedError('nonzero direct FEM flux requires an explicit load adapter')
                        selected = samplePatches == patch['patchId']
                        faces = np.asarray(mesh[f"patches/{patch['parentPatchUuid']}/{patchUid}"], int).ravel()
                        if (not np.any(selected) or np.any(sampleAreas[selected] <= 0)
                                or np.any(faces < 0) or np.any(faces >= len(surface))):
                            raise ValueError('invalid or empty exported surface patch')
                        physical = thermalPatch(sampleNodes[selected], sampleAreas[selected], surface[faces], len(nodes))
                        if not np.isclose(physical.area, patch['surfaceAreaM2'], rtol=1e-8):
                            raise ValueError('boundary samples must cover the full patch area')
                        result.patches[uid, patchUid] = physical
                ops = assembleLinearTetraThermalOperators(nodes, tetra, conductivity=conductivity,
                        density=rho, specificHeat=cp, heatSource=source, capacityMode=capacityMode)
                if not np.allclose(ops.elementVolumes, volumes, rtol=1e-8, atol=1e-16):
                    raise ValueError('exported element volumes disagree with assembled mesh')
                initial = float(region['initialConditions']['temperature'])
                result.components[name] = thermalModel(ops, referenceTemperature=initial)
                result.nodes[name], result.initialTemperatures[name] = nodes, initial
                result.audit['components'].append({'name': name, 'regionUuid': uid, 'nodeCount': len(nodes),
                    'elementCount': len(tetra), 'densityKgPerM3': rho, 'specificHeatJPerKgK': cp,
                    'conductivityWPerMK': conductivity.tolist(), 'capacityJPerK': float(ops.M.apply(np.ones(len(nodes))).sum()),
                    'volumePowerW': float(ops.Q.toNumpy().sum())})
            for uid, interface in thermal['femNetworkInterfaces'].items():
                if not interface.get('isActive', True) or interface.get('mode', 'network') != 'fem':
                    continue
                convection = interface['convectionUuid']
                if convection in spatialConvections:
                    raise ValueError('one convection cannot be driven by multiple FEM spatial fields')
                if not useSpatialConvection:
                    continue  # Explicit diagnostic ablation, never an automatic fallback.
                field = interface['htc3D']
                if len(field['timeSlices']) != 1 or float(field['timeSlices'][0]) != 0:
                    raise NotImplementedError('spatial HTC currently requires one static field at time zero')
                sourcePath = f"{field['tableUuid']}/mappingData/{field['timeSlices'][0]}"
                if sourcePath not in properties:
                    raise KeyError(f'spatial HTC mapping is missing from controlData.h5: {sourcePath}')
                side = interface['fem']
                name = regionNames[side['femUuid']]
                patch = result.patches[side['femUuid'], side['patchUuid']]
                h = np.asarray(properties[sourcePath], float).ravel()
                trace, g = _spatialConvectionTrace(patch, result.nodes[name], h)
                spatialConvections[convection] = (name, trace, g, {'source': sourcePath,
                    'interfaceUuid': uid, 'faceCount': len(h), 'minimumHtcWPerM2K': float(h.min()),
                    'maximumHtcWPerM2K': float(h.max()), 'zeroCoefficientFaceCount': int(np.count_nonzero(h == 0))})
            if set(regionNames) != {k for k,v in thermal['femRegions'].items() if v.get('isActive', True)}:
                raise ValueError('provide an exported component for every active FEM region')
            massNames = {}
            for uid, mass in thermal['thermalMasses'].items():
                if not mass.get('isActive', True):
                    continue
                name = f'mass_{uid}'
                material = mass['materialUuid']
                rho, cp = (_property(properties, material, prop) for prop in ('density', 'specificHeatCapacity'))
                capacity = rho * cp * float(mass['volume'])
                if not np.isfinite(capacity) or capacity <= 0:
                    raise ValueError('thermal mass capacity must be positive')
                power = _fixed(mass['heatSourceData'], 'thermal mass power')
                initial = float(mass['initialValue']['temperature'])
                ops = _operatorSet(matrixOperator(csr_matrix((1, 1)), 'scipy'), csr_matrix([[capacity]]), [power])
                result.components[name] = thermalModel(ops, referenceTemperature=initial)
                result.initialTemperatures[name], massNames[uid] = initial, name
                result.audit['components'].append({'name': name, 'nodeCount': 1, 'capacityJPerK': capacity,
                                                    'volumePowerW': power, 'materialName': control['material'][mass['materialType']+'s'][material]['name']})

        # Restore each contact once, using both sides at half weight. This yields
        # a symmetric positive-semidefinite exchange even on nonmatching meshes.
        for uid, contact in thermal['femFemTccInterfaces'].items():
            if not contact.get('isActive', True):
                continue
            if contact['tccSetting']['type'] != 'userInput' or contact.get('detailMode'):
                raise NotImplementedError('only constant userInput contact conductance is supported')
            sides = [contact['contactBody1'], contact['contactBody2']]
            if any(side.get('isHeatFluxInput') for side in sides):
                raise NotImplementedError('prescribed contact-side flux requires explicit decoupling')
            names = [regionNames[side['femUuid']] for side in sides]
            patches = [result.patches[side['femUuid'], side['patchUuid']] for side in sides]
            coefficient, radius = float(contact['tccSetting']['conductance']), float(contact['searchRadius'])
            if not np.isfinite(coefficient) or coefficient < 0 or not np.isfinite(radius) or radius <= 0:
                raise ValueError('invalid contact coefficient or search radius')
            audit = {'name': uid, 'primary': names[0], 'secondary': names[1],
                     'coefficientWPerM2K': coefficient, 'searchRadiusM': radius, 'sides': []}
            for side in (0, 1):
                primary, secondary = patches[side], patches[1-side]
                trace, distances = closestTriangleTrace(result.nodes[names[side]][primary.nodes],
                                      result.nodes[names[1-side]], secondary.triangles)
                covered = distances <= radius
                audit['sides'].append({'areaM2': primary.area, 'coveredAreaM2': float(primary.areas[covered].sum()),
                                      'uncoveredAreaM2': float(primary.areas[~covered].sum()),
                                      'maximumDistanceM': float(distances.max())})
                if np.any(covered):
                    indices = primary.nodes[covered]
                    nodal = csr_matrix((np.ones(len(indices)), (np.arange(len(indices)), indices)), shape=(len(indices), primary.order))
                    result.interfaces.append(thermalCouplingInterface(f'{uid}_side{side}', names[side], names[1-side],
                                nodal, trace[covered], coefficient * primary.areas[covered] / 2))
            if not any(v['coveredAreaM2'] > 0 for v in audit['sides']):
                raise ValueError(f'contact {uid} has no overlap within its original search radius')
            result.audit['contacts'].append(audit)
            if progress:
                progress(f'mapped contact {names[0]} / {names[1]}')

        # Union connected ports and both ports of each massless thermal surface.
        parents = {}
        def find(port):
            parents.setdefault(port, port)
            if parents[port] != port:
                parents[port] = find(parents[port])
            return parents[port]
        def join(a, b):
            parents[find(a)] = find(b)
        for port, connection in system['connections'].items():
            for target in connection['toPortUuids']:
                join(port, target)
        for uid, surface in thermal['thermalSurfaces'].items():
            if surface.get('isActive', True):
                if _fixed(surface['heatSourceData'], 'massless surface power') != 0:
                    raise NotImplementedError('nonzero massless surface sources require an explicit load adapter')
                join(f'{uid}@null&A', f'{uid}@null&B')
        endpoints = {}
        def register(port, endpoint):
            root = find(port)
            if root in endpoints:
                raise NotImplementedError('multiple physical endpoints share one ideal network node')
            endpoints[root] = endpoint
        for uid, name in massNames.items():
            register(f'{uid}@null&A', (name, csr_matrix([[1.]]), None))
        for uid, source in thermal['thermalSources'].items():
            if source.get('isActive', True):
                boundary = source['thermalBoundary']
                if boundary['boundaryType'] != 'temperatureBoundary':
                    raise NotImplementedError('only fixed-temperature network reservoirs are supported')
                register(f'{uid}@null&A', (None, None, _fixed(boundary['temperatureBoundary'], 'reservoir temperature')))
        for uid, interface in thermal['femNetworkInterfaces'].items():
            if not interface.get('isActive', True):
                continue
            side = interface['fem']
            name = regionNames[side['femUuid']]
            patch = result.patches[side['femUuid'], side['patchUuid']]
            trace = patch.averageTrace()
            register(f"{interface['network']['nodeUuid']}@null&A", (name, trace, None))
            if side.get('isHeatFluxInput'):
                heat = side['heatFlux']
                if heat['mode'] != 'table1D':
                    raise NotImplementedError('only fixed uniform prescribed patch loads are supported')
                value = _fixed(heat['heatFlux1D'], 'prescribed patch load')
                if heat['inputVariable'] == 'power':
                    power = value
                elif heat['inputVariable'] == 'heatFlux':
                    power = value * patch.area
                else:
                    raise NotImplementedError('unknown prescribed patch load variable')
                model = result.components[name]
                q = model.Q.toNumpy().ravel() + power * trace.toarray().ravel()
                result.components[name] = thermalModel(_operatorSet(model.K, model.M.toBackendData(), q, model.operators.elementVolumes),
                                                        referenceTemperature=model.referenceTemperature)
                result.audit['prescribedLoads'].append({'interface': uid, 'component': name, 'patchUuid': side['patchUuid'], 'powerW': power})
        for category in ('convections', 'conductions'):
            for uid, exchange in thermal[category].items():
                if not exchange.get('isActive', True):
                    continue
                info, geometry = exchange['thermalInfo'], exchange['geometry']
                if category == 'convections':
                    if geometry['geometryType'] != 'constHTC':
                        raise NotImplementedError('only constant-HTC network convection is supported')
                    g = _fixed(info['htc'], 'network HTC') * float(info['heatTransferArea']) * float(info['htcAdjust']) * float(info['areaAdjust'])
                else:
                    if geometry['conductionType'] != 'constantResistance' or geometry['isDiscretized']:
                        raise NotImplementedError('only nondiscretized constant thermal resistance is supported')
                    resistance = float(info['resistance'])
                    if not np.isfinite(resistance) or resistance <= 0:
                        raise ValueError('thermal resistance must be positive')
                    g = 1 / resistance
                if not np.isfinite(g) or g < 0:
                    raise ValueError('network conductance must be finite and nonnegative')
                sides = [endpoints[find(f'{uid}@null&{port}')] for port in ('A', 'B')]
                audit = {'name': uid, 'type': category, 'conductanceWPerK': g,
                    'primary': sides[0][0] or 'fixedReservoir', 'secondary': sides[1][0] or 'fixedReservoir',
                    'coefficientSource': 'networkConstant'}
                if uid in spatialConvections:
                    name, solidTrace, faceConductance, fieldAudit = spatialConvections[uid]
                    matching = [i for i, v in enumerate(sides) if v[0] == name]
                    if len(matching) != 1:
                        raise ValueError('spatial HTC solid does not match the network convection endpoint')
                    other = sides[1-matching[0]]
                    if other[0] not in massNames.values() or other[1].shape != (1, 1):
                        raise NotImplementedError('spatial FEM convection currently requires a lumped fluid endpoint')
                    audit.update(fieldAudit)
                    audit['placeholderConductanceWPerK'] = g
                    factor = float(info['htcAdjust']) * float(info['areaAdjust'])
                    if not np.isfinite(factor) or factor < 0:
                        raise ValueError('spatial HTC adjustment must be finite and nonnegative')
                    faceConductance = faceConductance * factor
                    audit['conductanceWPerK'] = float(faceConductance.sum())
                    audit['coefficientSource'] = 'femSpatialHtc'
                    fluidTrace = csr_matrix(np.ones((len(faceConductance), 1)))
                    result.interfaces.append(thermalCouplingInterface(uid, name, other[0], solidTrace, fluidTrace, faceConductance))
                    result.audit['network'].append(audit)
                    continue
                result.audit['network'].append(audit)
                if g == 0:
                    continue
                if sides[0][0] is not None and sides[1][0] is not None:
                    result.interfaces.append(thermalCouplingInterface(uid, sides[0][0], sides[1][0], sides[0][1], sides[1][1], [g]))
                elif sum(v[0] is None for v in sides) == 1:
                    (name, trace, _), (_, _, ambient) = sorted(sides, key=lambda v: v[0] is None)
                    model = result.components[name]
                    vector = trace.toarray().T
                    k = sumOperator([model.K, lowRankOperator(np.sqrt(g)*vector, backendName='scipy')])
                    q = model.Q.toNumpy().ravel() + g * ambient * vector.ravel()
                    result.components[name] = thermalModel(_operatorSet(k, model.M.toBackendData(), q, model.operators.elementVolumes),
                                                            referenceTemperature=model.referenceTemperature)
                else:
                    raise NotImplementedError('exchange between two fixed reservoirs is not a dynamic subsystem')
    return result
