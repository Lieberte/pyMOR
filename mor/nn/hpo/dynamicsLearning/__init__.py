import itertools
import math
from numbers import Integral, Real
import random
import sys

from mor.nn.hpo.baseHpo import baseHpo


class dynamicsHpo(baseHpo):
    def __init__(self, searchSpace, strategy='grid', maxTrials=None, randomSeed=0,
                 direction='minimize', continueOnError=False, **kwargs):
        super().__init__(hpoName='dynamicsHpo', **kwargs)
        if not isinstance(searchSpace, dict) or not searchSpace:
            raise ValueError('searchSpace must be a nonempty dict of parameter choices')
        self.searchSpace = {}
        for name, choices in searchSpace.items():
            if not isinstance(name, str) or not name or not isinstance(choices, (list, tuple)) or not choices:
                raise ValueError('each parameter must have a name and a nonempty list or tuple of choices')
            self.searchSpace[name] = tuple(choices)
        if strategy not in ('grid', 'random') or direction not in ('minimize', 'maximize'):
            raise ValueError('strategy must be grid/random and direction must be minimize/maximize')
        if maxTrials is not None and (not isinstance(maxTrials, Integral) or isinstance(maxTrials, bool) or maxTrials <= 0):
            raise ValueError('maxTrials must be a positive integer')
        if not isinstance(randomSeed, Integral) or isinstance(randomSeed, bool) or randomSeed < 0:
            raise ValueError('randomSeed must be a non-negative integer')
        self.totalTrials = math.prod(len(choices) for choices in self.searchSpace.values())
        if self.totalTrials > sys.maxsize:
            raise ValueError('searchSpace has too many combinations')
        self.maxTrials = min(self.totalTrials, self.totalTrials if maxTrials is None else int(maxTrials))
        self.strategy, self.direction, self.randomSeed = strategy, direction, int(randomSeed)
        self.continueOnError = continueOnError
        self.lastTrials = []

    def _parameters(self):
        names, choices = list(self.searchSpace), list(self.searchSpace.values())
        if self.strategy == 'grid':
            for values in itertools.islice(itertools.product(*choices), self.maxTrials):
                yield dict(zip(names, values))
        else:
            for index in random.Random(self.randomSeed).sample(range(self.totalTrials), self.maxTrials):
                values = []
                for options in reversed(choices):
                    index, position = divmod(index, len(options))
                    values.append(options[position])
                yield dict(zip(names, reversed(values)))

    def optimize(self, objective) -> dict:
        import copy
        if not callable(objective):
            raise ValueError('objective must be callable and return a finite scalar score')
        self.lastTrials = []
        best = None
        for parameters in self._parameters():
            trial = {'params': copy.deepcopy(parameters), 'status': 'failed'}
            try:
                score = objective(copy.deepcopy(parameters))
                if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                    raise ValueError('objective must return a finite real scalar score')
                trial.update(score=float(score), status='completed')
            except Exception as error:
                trial['error'] = f'{type(error).__name__}: {error}'
                self.lastTrials.append(trial)
                if not self.continueOnError: raise
                continue
            self.lastTrials.append(trial)
            if best is None or (trial['score'] < best['score'] if self.direction == 'minimize' else trial['score'] > best['score']):
                best = trial
        if best is None:
            raise ValueError('all HPO trials failed; inspect lastTrials')
        return copy.deepcopy({'bestParams': best['params'], 'bestScore': best['score'],
                              'trials': self.lastTrials, 'strategy': self.strategy, 'direction': self.direction})
