import numpy as np
from collections import deque, OrderedDict
import os

import time
import psutil
# import resource
# import objgraph


PROFILE_SCOPE = []
PROFILE_SCOPE_CACHE = []


def mem_check_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


class Profiler:
    def __init__(self, name="profiler", time=True, memory=False, cache=False):
        self.name = name
        self.time = time
        self.memory = memory
        self.cache = cache

    def __enter__(self):
        global PROFILE_SCOPE
        global PROFILE_SCOPE_CACHE
        if not self.cache:
            PROFILE_SCOPE_CACHE.append(OrderedDict())
        PROFILE_SCOPE.append(self.name)
        if self.time:
            self.start = time.clock()
        if self.memory:
            self.start_mem = mem_check_usage()
            # objgraph.show_growth()
        return self

    def __exit__(self, *args):
        global PROFILE_SCOPE
        global PROFILE_SCOPE_CACHE
        if self.time:
            self.end = time.clock()
            self.interval = self.end - self.start
            if self.cache:
                key = "[{}]".format("][".join(PROFILE_SCOPE))
                PROFILE_SCOPE_CACHE[-1][key] = PROFILE_SCOPE_CACHE[-1].get(key, 0) + self.interval # accumulate timing
            else:
                print("TIMING [{}]: {}".format("][".join(PROFILE_SCOPE), self.interval))
                dd = PROFILE_SCOPE_CACHE.pop()
                for kk, vv in dd.items():
                    print("TIMING-CACHE {} {}".format(kk, vv))
        if self.memory:
            # import ipdb; ipdb.set_trace()
            self.end_mem = mem_check_usage()
            self.interval_mem = self.end_mem - self.start_mem
            print("MEMORY [{}]: i{} s{} e{}".format("][".join(PROFILE_SCOPE), self.interval_mem, self.start_mem, self.end_mem))
            # objgraph.show_growth()
        PROFILE_SCOPE.pop()


class GenericClass(object):
    def __init__(self, **kwargs):
        super(GenericClass, self).__init__()
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def __repr__(self):
        s = "{}"
        return s.format(self.__dict__)


class Args(GenericClass): pass


class EncodeArgs(GenericClass): pass


class Vocab(GenericClass): pass


class Example(GenericClass): pass


def time_per_token(num_tokens, total_time):
    return sum(total_time) / float(sum(num_tokens))


class Accumulator(object):
    """Accumulator. Makes it easy to keep a trailing list of statistics."""

    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.cache = dict()

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.maxlen)).append(val)

    def get(self, key, clear=True):
        ret = self.cache.get(key, [])
        if clear:
            try:
                del self.cache[key]
            except:
                pass
        return ret

    def get_avg(self, key, clear=True):
        return np.array(self.get(key, clear)).mean()


class MetricsLogger(object):
    """MetricsLogger."""

    def __init__(self, metrics_path):
        self.metrics_path = metrics_path

    def Log(self, key, val, step):
        log_path = os.path.join(self.metrics_path, key) + ".metrics"
        with open(log_path, 'a') as f:
            f.write("{} {}\n".format(step, val))


class EvalReporter(object):
    def __init__(self):
        self.batches = []

    def save_batch(self, preds, target, example_ids, output, sent1_transitions=None, sent2_transitions=None):
        sent1_transitions = sent1_transitions if sent1_transitions is not None else [None] * len(example_ids)
        sent2_transitions = sent2_transitions if sent2_transitions is not None else [None] * len(example_ids)
        batch = [preds.view(-1), target.view(-1), example_ids, output, sent1_transitions, sent2_transitions]
        self.batches.append(batch)

    def write_report(self, filename):
        with open(filename, 'w') as f:
            for b in self.batches:
                for bb in zip(*b):
                    pred, truth, eid, output, sent1_transitions, sent2_transitions = bb
                    report_str = "{eid} {correct} {truth} {pred} {output}"
                    if sent1_transitions is not None:
                        report_str += " {sent1_transitions}"
                    if sent2_transitions is not None:
                        report_str += " {sent2_transitions}"
                    report_str += "\n"
                    report_dict = {
                        "eid": eid,
                        "correct": truth == pred,
                        "truth": truth,
                        "pred": pred,
                        "output": " ".join([str(o) for o in output]),
                        "sent1_transitions": '{}'.format("".join(str(t) for t in sent1_transitions)) if sent1_transitions is not None else None,
                        "sent2_transitions": '{}'.format("".join(str(t) for t in sent2_transitions)) if sent2_transitions is not None else None,
                    }
                    f.write(report_str.format(**report_dict))


def recursively_set_device(inp, gpu):
    if hasattr(inp, 'keys'):
        for k in inp.keys():
            inp[k] = recursively_set_device(inp[k], gpu)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, gpu) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, gpu) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if gpu >= 0:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp
