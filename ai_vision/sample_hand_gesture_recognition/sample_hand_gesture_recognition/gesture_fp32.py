# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Stage-3 gesture classifier: an FP32 numpy implementation that runs locally
# inside the node process.
#
# Why not on the NPU:
#   The w8a8-quantized canned_gesture_classifier context binary loses too much
#   accuracy (almost everything except Thumb_Up collapses to None). The
#   classifier itself is a tiny MLP (64 -> 128 -> ... -> 8, an embedder plus a
#   classification head), so running it in FP32 with numpy is negligible in cost
#   yet fully accurate. This matches the architecture of
#   qrb_ros_samples/ai_vision/sample_hand_detection -- "large models on the NPU,
#   post-processing math in numpy inside the node" (the palm/landmark blaze
#   decode, NMS and sigmoid are likewise done in numpy inside the node).
#
# The weights come from Qualcomm's official AI Hub assets (downloadable from the
# public qai-hub bucket, see config.yaml):
#   gesture_embedder.pth   -> inner_mlp.* + bn_out.* + fc_out.*
#   gesture_classifier.pth -> gamma / beta + fc.*
# The network structure matches quic/ai-hub-models:
#   src/qai_hub_models/models/mediapipe_hand_gesture/model.py
#     :: CannedGestureClassifier (symmetric inner(hand)+inner(mirrored_hand))

import io
import os
import pickle
import zipfile

import numpy as np

from gesture_classifier import GESTURE_LABELS

# BatchNorm eps in eval mode (matches the official torch default).
_BN_EPS = 1e-3


# --------------------------------------------------------------------------
# torch-free .pth reader: a torch state_dict .pth is essentially a zip
#   (data.pkl is a pickle, data/<key> holds the raw bytes of each tensor).
#   The board has no torch, so we decode name -> np.ndarray directly with
#   numpy + the standard library.
# --------------------------------------------------------------------------
_DTYPE_MAP = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64,
    'LongStorage': np.int64, 'IntStorage': np.int32,
    'HalfStorage': np.float16, 'ByteStorage': np.uint8,
}


class _Stub:
    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        return ('CALL', self.name, a, k)


def _load_state_dict(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split('/')[0]

    def read_storage(key):
        return z.read(f'{root}/data/{key}')

    class _Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2':
                def rebuild(storage, offset, size, stride, rg, bh, *extra):
                    key, dtype = storage
                    arr = np.frombuffer(read_storage(key), dtype=dtype)
                    n = int(np.prod(size)) if size else arr.size
                    arr = arr[offset:offset + n]
                    if size:
                        arr = arr.reshape(size)
                    return np.array(arr)
                return rebuild
            if name == 'OrderedDict':
                from collections import OrderedDict
                return OrderedDict
            if 'Storage' in name:
                return ('STORAGETYPE', _DTYPE_MAP.get(name, np.float32))
            return _Stub(f'{mod}.{name}')

        def persistent_load(self, pid):
            # pid = ('storage', storagetype, key, location, numel)
            typ = pid[1]
            dt = typ[1] if isinstance(typ, tuple) and typ[0] == 'STORAGETYPE' else np.float32
            return (str(pid[2]), dt)

    return _Unpickler(io.BytesIO(z.read(f'{root}/data.pkl'))).load()


# --------------------------------------------------------------------------
class GestureClassifier:
    """FP32 numpy canned gesture classifier.

    classify(x64_a, x64_b) takes two [1,64] inputs (hand / mirrored_hand, already
    normalized by preprocess_hand_x64) and returns the 8-class softmax
    probabilities.
    """

    def __init__(self, model_path,
                 embedder='gesture_embedder.pth',
                 classifier='gesture_classifier.pth'):
        emb = _load_state_dict(os.path.join(model_path, embedder))
        cls = _load_state_dict(os.path.join(model_path, classifier))
        self.sd = {**emb, **cls}

    # ---- basic ops ----
    @staticmethod
    def _lin(x, w, b):
        return x @ w.T + b

    def _bn(self, x, p):
        s = self.sd
        return (x - s[p + '.running_mean']) / np.sqrt(s[p + '.running_var'] + _BN_EPS) \
            * s[p + '.weight'] + s[p + '.bias']

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    # ---- the embedder's residual MLP (7 residual blocks) ----
    def _inner(self, x):
        s = self.sd
        z0 = self._lin(x, s['inner_mlp.fc0.weight'], s['inner_mlp.fc0.bias'])
        h = self._silu(self._bn(z0, 'inner_mlp.bn0'))
        acc = z0 + self._lin(h, s['inner_mlp.fcs.0.weight'], s['inner_mlp.fcs.0.bias'])
        for i in range(1, 6):
            hi = self._silu(self._bn(acc, f'inner_mlp.bns_after_add.{i - 1}'))
            acc = acc + self._lin(hi, s[f'inner_mlp.fcs.{i}.weight'],
                                  s[f'inner_mlp.fcs.{i}.bias'])
        h6 = self._silu(self._bn(acc, 'inner_mlp.bns_after_add.5'))
        return acc + self._lin(h6, s['inner_mlp.fcs.6.weight'], s['inner_mlp.fcs.6.bias'])

    def classify(self, x64_a, x64_b):
        """Return (gesture_id, gesture_name, scores[8])."""
        s = self.sd
        a = self._inner(np.asarray(x64_a, dtype=np.float32).reshape(1, 64))
        b = self._inner(np.asarray(x64_b, dtype=np.float32).reshape(1, 64))
        h = self._silu(self._bn(a + b, 'bn_out'))
        emb = self._lin(h, s['fc_out.weight'], s['fc_out.bias'])
        x = np.maximum(emb * s['gamma'] + s['beta'], 0.0)   # gamma/beta + ReLU before the head
        logits = self._lin(x, s['fc.weight'], s['fc.bias'])
        e = np.exp(logits - logits.max(-1, keepdims=True))
        scores = (e / e.sum(-1, keepdims=True)).reshape(-1)
        gesture_id = int(np.argmax(scores))
        return gesture_id, GESTURE_LABELS[gesture_id], scores
