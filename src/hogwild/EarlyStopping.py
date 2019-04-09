import math
import numpy as np

class EarlyStopping():
    ''' Class for early stopping when the loss stops improving '''
    def __init__(self, persistence, epsilon):
        self.persistence = persistence
        self.window = []
        self.epsilon = epsilon

    def stopping_criterion(self, current_loss):
        ''' Returns True if the loss has stopped improving since persistence
        number of epochs. '''
        window_size = len(self.window)
        if window_size < self.persistence * 2:
            return False
        del self.window[0]
        self.window.append(current_loss)
        recent_window_mean = sum(self.window[-self.persistence:]) / self.persistence
        old_window_mean = sum(self.window[0:self.persistence]) / self.persistence
        if recent_window_mean > old_window_mean - self.epsilon:
            return True
        return False
