import io
import json
import zipfile

import h5py
import numpy as np
import pytest
from scipy import linalg

from mor.assembly import loadThermalArchive
from mor.assembly.surfaceMapping import closestTriangleTrace


def _h5(datasets):
    stream = io.BytesIO()
    with h5py.File(stream, 'w') as f:
        for name, values in datasets.items():
            f.create_dataset(name, data=values)
    return stream.getvalue()


def _archive(tmp_path, *, badSource=False, spatial=False):
    nodes = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    faces = np.array([[0,1,2]])
    tetra = np.array([[0,1,2,3]])
    table = lambda value: {'type':'fixedValue','fixedValue':value}
    heat = {'mode':'table1D','inputVariable':'power','heatSource1D':table(2.)}
    if badSource:
        heat['heatSource1D']['type'] = 'rawTable'
    region = {'isActive':True,'meshUuid':'mesh','materialUuid':'solid',
              'initialConditions':{'temperature':300.},
              'patches':{'parent':{'children':{'face':{'boundaryConditions':{'heatFlux':None}}}}},
              'volumePatches':{'v':{'children':{'v':{'volumetricGroups':{'heatSource':heat}}}}}}
    thermal = {'femRegions':{'solid':region},'femFemTccInterfaces':{},
        'thermalMasses':{'oil':{'materialUuid':'fluid','materialType':'fluid','volume':1.,
            'heatSourceData':table(1.),'initialValue':{'temperature':300.}}},
        'thermalSurfaces':{'wet':{'heatSourceData':table(0.)},'air':{'heatSourceData':table(0.)}},
        'thermalSources':{'ambient':{'thermalBoundary':{'boundaryType':'temperatureBoundary',
                                         'temperatureBoundary':table(290.)}}},
        'femNetworkInterfaces':{'input':{'network':{'nodeUuid':'wet'},
            'fem':{'femUuid':'solid','patchUuid':'face','isHeatFluxInput':True,
                   'heatFlux':{'mode':'table1D','inputVariable':'power','heatFlux1D':table(3.)}}}},
        'convections':{'wetConvection':{'geometry':{'geometryType':'constHTC'},
            'thermalInfo':{'htc':table(4.),'heatTransferArea':.5,'areaAdjust':1.,'htcAdjust':1.}},
            'airConvection':{'geometry':{'geometryType':'constHTC'},
            'thermalInfo':{'htc':table(2.),'heatTransferArea':.5,'areaAdjust':1.,'htcAdjust':1.}}},
        'conductions':{}}
    connections = {
        'wetConvection@null&A':{'toPortUuids':['oil@null&A']},
        'wetConvection@null&B':{'toPortUuids':['wet@null&A']},
        'airConvection@null&A':{'toPortUuids':['air@null&A']},
        'airConvection@null&B':{'toPortUuids':['wet@null&B']},
        'air@null&B':{'toPortUuids':['ambient@null&A']}}
    control = {'rootSystem':{'children':{},'thermalComponents':thermal,'connections':connections},
               'material':{'fluids':{'fluid':{'name':'test oil'}},
                           'solids':{'solid':{'thermalConductivity':{'type':'isotropic',
                                                                     'isotropic':table(1.)}}}}}
    materials = {f'{m}/{p}/isotropic/fixedValue':[v] for m in ('solid','fluid')
                 for p,v in [('density',2.),('specificHeatCapacity',3.)]}
    materials['solid/thermalConductivity/isotropic/fixedValue']=[1.]
    mesh = {'nodes':nodes,'connectivityVolume':tetra,'connectivitySurface':faces,
            'patches/parent/face':[0],'volumes/v':[0]}
    if spatial:
        interface=thermal['femNetworkInterfaces']['input']
        interface.update(mode='fem',convectionUuid='wetConvection',
                         htc3D={'tableUuid':'htcField','timeSlices':['0.000000']})
        materials['htcField/mappingData/0.000000']=[8.]
        thermal['convections']['wetConvection']['thermalInfo']['htc']=table(0.)
    sta = tmp_path/'source.sta'
    with zipfile.ZipFile(sta,'w') as z:
        z.writestr('controlData.json',json.dumps(control))
        z.writestr('controlData.h5',_h5(materials))
        z.writestr('mesh/mesh.samesh',_h5(mesh))
    folder=tmp_path/'component';folder.mkdir()
    np.savez(folder/'data.npz',nodesMeters=nodes,tetrahedra=tetra,surfaceTriangles=faces,
        elementVolumesM3=[1/6],conductivityTensorWPerMK=np.eye(3),boundarySampleNodeIndices=[0,1,2],
        boundaryWeightsM2=np.full(3,1/6),boundaryPatchIds=np.zeros(3,int),
        # Poison solved fields: loading must neither depend on them nor use them.
        nodeTemperatureK=np.full(4,np.nan),boundaryHeatFluxWPerM2=np.full(3,np.nan))
    manifest={'regionUuid':'solid','dataFile':'data.npz','metadata':{'surfacePatches':[
        {'patchUuid':'face','parentPatchUuid':'parent','patchId':0,'surfaceAreaM2':.5}]}}
    (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return sta,folder


def test_archive_sources_network_and_autonomous_step_match_independent_full_equations(tmp_path):
    sta,folder=_archive(tmp_path)
    assembly=loadThermalArchive(sta,{'body':folder})
    assert len(assembly.interfaces)==1
    assert assembly.components['body'].Q.toNumpy().sum()==pytest.approx(295.) # 2+3+290 W
    assert assembly.components['mass_oil'].M.toNumpy()[0,0]==pytest.approx(6.)
    assert assembly.audit['resultFieldsUsedAsInputs'] is False
    rom=assembly.reduce({'body':np.eye(4),'mass_oil':np.ones((1,1))})
    trace=np.array([1/3,1/3,1/3,0.])
    # Independent tetra M/K, face-average ambient and wet exchange.
    mass=linalg.block_diag((np.ones((4,4))+np.eye(4))/20,[[6.]])
    gradients=np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])
    k=linalg.block_diag(gradients@gradients.T/6+np.outer(trace,trace),[[0.]])
    jump=np.r_[trace,-1.]
    k+=2*np.outer(jump,jump)
    q=np.r_[np.full(4,.5)+293*trace,1.]
    initial=np.full(5,300.)
    expected=linalg.solve(mass+.2*k,mass@initial+.2*q)
    result=rom.solveTransient(assembly.initialTemperatures,[0.,.2])
    np.testing.assert_allclose(np.r_[result.temperatures('body')[-1],result.temperatures('mass_oil')[-1]],expected,atol=1e-10)
    assert abs(result.diagnostics['residualW'][0])<1e-9
    # Includes 6 W sources and heat lost to the 290 K reservoir.
    assert result.diagnostics['externalPowerW'][0]==pytest.approx(6-(trace@expected[:4]-290.))


def test_archive_rejects_nonconstant_sources(tmp_path):
    sta,folder=_archive(tmp_path,badSource=True)
    with pytest.raises(NotImplementedError,match='fixedValue'):
        loadThermalArchive(sta,{'body':folder})


def test_spatial_htc_overrides_zero_placeholder_and_integrates_consistent_face_mass(tmp_path):
    sta,folder=_archive(tmp_path,spatial=True)
    assembly=loadThermalArchive(sta,{'body':folder})
    interface=assembly.interfaces[0]
    assert interface.name=='wetConvection'
    assert interface.conductance.sum()==pytest.approx(4.) # 8 W/(m² K) over 0.5 m²
    trace=interface.primaryTrace.toarray()
    expected=np.zeros((4,4));expected[:3,:3]=(np.ones((3,3))+np.eye(3))/3
    np.testing.assert_allclose(trace.T@(interface.conductance[:,None]*trace),expected)
    np.testing.assert_allclose(trace.T@interface.conductance,[4/3,4/3,4/3,0.])
    audit=next(v for v in assembly.audit['network'] if v['name']=='wetConvection')
    assert audit['coefficientSource']=='femSpatialHtc'
    assert audit['placeholderConductanceWPerK']==0.
    rom=assembly.reduce({'body':np.eye(4),'mass_oil':np.ones((1,1))})
    result=rom.solveTransient(assembly.initialTemperatures,[0.,.2])
    assert abs(result.diagnostics['residualW'][0])<1e-9
    # The fluid is physical; fixed-interface diagnostics need no per-face flux evaluation.
    assert not rom._fixedInterfaces
    legacy=loadThermalArchive(sta,{'body':folder},useSpatialConvection=False)
    assert not legacy.interfaces


def test_closest_triangle_interpolates_planes_and_clamps_edges_without_radius_changes():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    trace,distance=closestTriangleTrace([[.2,.3,.1],[2.,0.,0.]],nodes,np.array([[0,1,2]]))
    np.testing.assert_allclose(trace.toarray(),[[.5,.2,.3],[0.,1.,0.]])
    np.testing.assert_allclose(distance,[.1,1.])
    np.testing.assert_allclose(trace@np.ones(3),1.)
    with pytest.raises(ValueError,match='degenerate'):
        closestTriangleTrace([[0.,0.,0.]],nodes,np.array([[0,0,1]]))
