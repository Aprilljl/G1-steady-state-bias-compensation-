"""Policies for the 12-DoF baseline: TorchScript exports (deploy_mujoco style) and rsl_rl 1.0.x checkpoints
(MLP ActorCritic, LSTM / GRU ActorCriticRecurrent) in pure torch. Python 3.8."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))  # thesis_eval/ for policy_arch
from policy_arch import infer_arch  # noqa: E402

ACTIVATIONS = {"elu": nn.ELU, "selu": nn.SELU, "relu": nn.ReLU, "crelu": nn.CELU, "lrelu": nn.LeakyReLU,
               "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}  # rsl_rl 1.0 get_activation names


class JitPolicy(object):
    kind = "jit"

    def __init__(self, path):
        self.m = torch.jit.load(str(path), map_location="cpu").eval()
        self.path = str(path)

    def reset(self):
        if hasattr(self.m, "reset_memory"):
            self.m.reset_memory()

    @torch.no_grad()
    def act(self, obs):
        out = self.m(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0))
        return out.squeeze(0).numpy().astype(np.float64), {}


class CkptPolicy(object):
    """rsl_rl checkpoint model_*.pt, deterministic mean action (act_inference). Observation normalisation does not exist
    in rsl_rl 1.0.x. For recurrent nets the hidden state persists across steps and is zeroed by reset()."""
    kind = "ckpt"

    def __init__(self, path, activation="elu"):
        try:
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:  # old torch without weights_only
            ck = torch.load(str(path), map_location="cpu")
        sd = ck["model_state_dict"]
        self.arch = infer_arch(sd)
        a = self.arch
        self.path = str(path)
        self.iteration = ck.get("iter")
        keys = sorted((k for k in sd if k.startswith("actor.") and k.endswith(".weight")),
                      key=lambda k: int(k.split(".")[1]))
        layers = []
        for i, k in enumerate(keys):
            w = sd[k]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(sd[k.replace("weight", "bias")])
            layers.append(lin)
            if i < len(keys) - 1:
                layers.append(ACTIVATIONS[activation]())
        self.actor = nn.Sequential(*layers).eval()
        self.rnn = None
        if a["rnn_type"] is not None:
            cls = nn.LSTM if a["rnn_type"] == "lstm" else nn.GRU
            self.rnn = cls(a["num_actor_obs"], a["rnn_hidden_size"], a["rnn_num_layers"])
            sub = dict((k[len("memory_a.rnn."):], v) for k, v in sd.items() if k.startswith("memory_a.rnn."))
            self.rnn.load_state_dict(sub)
            self.rnn.eval()
        self.hidden = None
        self.kind = "ckpt_" + (a["rnn_type"] or "mlp")

    def reset(self):
        self.hidden = None

    @torch.no_grad()
    def act(self, obs):
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if self.rnn is not None:
            out, self.hidden = self.rnn(x.unsqueeze(0), self.hidden)  # (seq=1, batch=1, in)
            x = out.squeeze(0)
        return self.actor(x).squeeze(0).numpy().astype(np.float64), {}


def load_policy(spec):
    """spec: dict(kind='jit'|'ckpt', path=..., activation='elu')."""
    path = os.path.expanduser(spec["path"])
    if spec["kind"] == "jit":
        return JitPolicy(path)
    return CkptPolicy(path, spec.get("activation", "elu"))
