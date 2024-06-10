# MIT License

# Copyright (c) 2024 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import time
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def _convert_seconds(seconds):
    '''
    convert seconds to days, hours, minutes, seconds.
    '''
    # Calculate days
    days = int(seconds // (24 * 3600))
    seconds %= (24 * 3600)

    # Calculate hours
    hours = int(seconds // 3600)
    seconds %= 3600

    # Calculate minutes
    minutes = int(seconds // 60)
    seconds %= 60

    return days, hours, minutes, int(seconds)


class SingleLoss:
    def __init__(self, name: str, writer: SummaryWriter):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.writer = writer

    def add_scalar(self, val, step=None):
        if step is None:
            step = len(self.loss_step)
        self.loss_step.append(val)
        self.loss_epoch_tmp.append(val)
        self.writer.add_scalar('Train/step_' + self.name, val, step)

    def epoch(self, step=None):
        if step is None:
            step = len(self.loss_epoch)
        loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        self.loss_epoch_tmp = []
        self.loss_epoch.append(loss_avg)
        self.writer.add_scalar('Train/epoch_' + self.name, loss_avg, step)

    def save(self, path):
        loss_step = np.array(self.loss_step)
        loss_epoch = np.array(self.loss_epoch)
        np.save(path + self.name + '_step.npy', loss_step)
        np.save(path + self.name + '_epoch.npy', loss_epoch)


class SingleVideo:
    def __init__(self, name: str, writer: SummaryWriter):
        self.name = name
        self.writer = writer
        self.video_step = []

    def add_video(self, val, step=None):
        if step is None:
            step = len(self.video_step)
        self.video_step.append('')
        self.writer.add_video('Train/step_' + self.name, val, step)


class LossRecorder:
    '''
    the loss recorder for tensorboard

    initialize:
        loss_recoder = LossRecorder(writer)
    '''

    def __init__(self, writer: SummaryWriter):
        self.losses: Dict[str, SingleLoss] = {}
        self.videos = {}
        self.writer = writer
        self.start_time = time.time()

    def add_scalar(self, name: str, val, step: Union[int, None] = None):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer)
        self.losses[name].add_scalar(val, step)

    def add_video(self, name, val, step=None):
        ''' 
        add video to tensorboard
        '''
        if name not in self.videos:
            self.videos[name] = SingleVideo(name, self.writer)
        self.videos[name].add_video(val, step)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        for loss in self.losses.values():
            loss.save(path)

    def start_timer(self):
        '''
        start the timer
        '''
        self.start_time = time.time()

    def end_timer(self):
        '''
        end the timer and print the time
        '''
        d, h, m, s = _convert_seconds(time.time() - self.start_time)
        print(f"D {d} - H {h} - M {m} - S {s}")
