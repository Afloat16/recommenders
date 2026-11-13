# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
# Original codes are from
# https://github.com/kang205/SASRec/blob/master/sampler.py

import numpy as np
from multiprocessing import Process, Queue


def random_neq(left, right, s):
    t = np.random.randint(left, right)
    while t in s:
        t = np.random.randint(left, right)
    return t


def _valid_users(user_train, usernum, itemnum):
    """Find users that can supply both a sequence target and an unseen item."""
    users = [
        user
        for user, history in user_train.items()
        if 1 <= user <= usernum
        and len(history) > 1
        and len({item for item in history if 1 <= item <= itemnum}) < itemnum
    ]
    if not users:
        raise ValueError(
            "SASRec sampling requires a user with at least two interactions "
            "and an unseen item for negative sampling"
        )
    return users


def sample_function(
    user_train,
    usernum,
    itemnum,
    batch_size,
    maxlen,
    result_queue,
    seed,
    valid_users=None,
):
    """Batch sampler that creates a sequence of negative items based on the
    original sequence of items (positive) that the user has interacted with.

    Args:
        user_train (dict): dictionary of training exampled for each user
        usernum (int): number of users
        itemnum (int): number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        result_queue (multiprocessing.Queue): queue for storing sample results
        seed (int): seed for random generator
        valid_users (list | None): eligible users, computed when not supplied
    """

    if valid_users is None:
        valid_users = _valid_users(user_train, usernum, itemnum)

    def sample():

        user = valid_users[np.random.randint(len(valid_users))]

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(seed)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """Sampler object that creates an iterator for feeding batch data while training.

    Users without a sequence target or an unseen negative item are skipped.
    Construction raises ValueError if no user can produce a training pair.

    Attributes:
        User: dict, all the users (keys) with items as values
        usernum: integer, total number of users
        itemnum: integer, total number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        n_workers (int): number of workers for parallel execution
    """

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        valid_users = _valid_users(User, usernum, itemnum)
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9),
                        valid_users,
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
