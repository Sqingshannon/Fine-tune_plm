import numpy as np

class EarlyStopper(object):
    def __init__(self,
            patience=None, eval_freq=1, best_score=None,
            delta=1e-9, higher_better=True):
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_score = best_score
        self.delta = delta
        self.higher_better = higher_better
        self.counter = 0
        self.early_stop = False

    def not_improved(self, val_score):
        if np.isnan(val_score):
            return True
        if self.higher_better:
            return val_score < self.best_score + self.delta
        else:
            return val_score > self.best_score - self.delta

    def update(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            is_best = True
        elif self.not_improved(val_score):
            self.counter += self.eval_freq
            if (self.patience is not None) and (self.counter > self.patience):
                self.early_stop = True
            is_best = False
        else:
            self.best_score = val_score
            self.counter = 0
            is_best = True
        return is_best