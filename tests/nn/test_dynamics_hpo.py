import numpy as np
import pytest
import torch

from mor.nn.hpo import dynamicsHpo
from mor.nn.reductors.dynamicsLearningPipelineTorch import runDynamicsLearningTorch
from tests.nn.test_training_pipeline import configFor


def test_grid_and_maximization_find_expected_best_and_keep_trial_snapshots():
    hpo = dynamicsHpo({'x': [1, 2, 3], 'y': [0, 1]}, direction='maximize')
    result = hpo.optimize(lambda p: p['x'] - p['y'])
    assert result['bestParams'] == {'x': 3, 'y': 0}
    assert result['bestScore'] == 3
    assert len(result['trials']) == 6
    result['trials'][0]['params']['x'] = -100
    assert hpo.lastTrials[0]['params']['x'] == 1


def test_random_search_is_reproducible_without_duplicate_trials_or_global_rng_changes():
    hpo = dynamicsHpo({'x': list(range(10)), 'y': [0, 1]}, strategy='random', maxTrials=8, randomSeed=7)
    state = np.random.get_state()
    first = hpo.optimize(lambda p: (p['x'] - 3) ** 2 + p['y'])
    second = hpo.optimize(lambda p: (p['x'] - 3) ** 2 + p['y'])
    assert first == second
    assert len({tuple(t['params'].values()) for t in first['trials']}) == 8
    assert np.array_equal(np.random.get_state()[1], state[1])


def test_hpo_error_policy_and_nonfinite_scores():
    hpo = dynamicsHpo({'x': [0, 1]}, continueOnError=True)
    result = hpo.optimize(lambda p: np.nan if p['x'] == 0 else 1.)
    assert result['bestParams'] == {'x': 1}
    assert result['trials'][0]['status'] == 'failed'
    with pytest.raises(ValueError, match='all HPO'): hpo.optimize(lambda p: np.inf)
    strict = dynamicsHpo({'x': [0]})
    with pytest.raises(RuntimeError, match='objective'):
        strict.optimize(lambda p: (_ for _ in ()).throw(RuntimeError('objective failed')))
    assert strict.lastTrials[0]['status'] == 'failed'


@pytest.mark.parametrize('options', [dict(searchSpace={}), dict(searchSpace={'x': []}),
    dict(searchSpace={'x': [1]}, maxTrials=0), dict(searchSpace={'x': [1]}, randomSeed=-1),
    dict(searchSpace={'x': [1]}, strategy='bad')])
def test_hpo_rejects_invalid_search_request(options):
    with pytest.raises(ValueError): dynamicsHpo(**options)


def test_hpo_runs_actual_dynamics_training_and_preserves_torch_rng():
    state = torch.get_rng_state().clone()
    def objective(parameters):
        config = configFor('mambaDynamicsTorch')
        config.optimizer.learningRate = parameters['learningRate']
        values = np.arange(24).reshape(4, 3, 2) / 24
        result = runDynamicsLearningTorch(config, values, values)
        return result['validationResult']['mse']
    result = dynamicsHpo({'learningRate': [0.001, 0.01]}, maxTrials=2).optimize(objective)
    assert np.isfinite(result['bestScore'])
    assert len(result['trials']) == 2
    assert torch.equal(torch.get_rng_state(), state)
