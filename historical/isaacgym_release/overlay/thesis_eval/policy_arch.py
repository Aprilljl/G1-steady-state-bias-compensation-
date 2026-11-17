"""Infer the rsl_rl 1.0.x actor-critic architecture from a checkpoint state_dict."""
import hashlib
import re

import torch

_MLP_KEY = re.compile(r"^(actor|critic)\.(\d+)\.(weight|bias)$")
_RNN_IH_KEY = re.compile(r"^(memory_[ac])\.rnn\.weight_ih_l(\d+)$")


def _mlp_linear_shapes(state_dict, prefix):
    """Return [(out_dim, in_dim), ...] of `<prefix>.{i}.weight`, sorted by integer i."""
    idx = {}
    for key, value in state_dict.items():
        m = _MLP_KEY.match(key)
        if m and m.group(1) == prefix and m.group(3) == "weight":
            if value.dim() != 2:
                raise ValueError("%s is not a 2-D Linear weight: %s" % (key, tuple(value.shape)))
            idx[int(m.group(2))] = tuple(value.shape)
    if not idx:
        raise ValueError("no '%s.{i}.weight' keys found in state_dict" % prefix)
    shapes = [idx[i] for i in sorted(idx)]
    for prev, cur in zip(shapes[:-1], shapes[1:]):
        if prev[0] != cur[1]:
            raise ValueError("%s layer shapes are not chained: %s -> %s" % (prefix, prev, cur))
    return shapes


def _rnn_info(state_dict, name):
    """Return (rnn_type, hidden_size, num_layers, input_size) of `<name>.rnn`, or None."""
    layers = sorted(int(m.group(2)) for m in
                    (_RNN_IH_KEY.match(k) for k in state_dict) if m and m.group(1) == name)
    if not layers:
        return None
    if layers != list(range(len(layers))):
        raise ValueError("%s.rnn layer indices are not contiguous: %s" % (name, layers))
    w_ih = state_dict["%s.rnn.weight_ih_l0" % name]
    w_hh_key = "%s.rnn.weight_hh_l0" % name
    if w_hh_key not in state_dict:
        raise ValueError("missing key %s" % w_hh_key)
    hidden = state_dict[w_hh_key].shape[1]
    ratio = w_ih.shape[0] / float(hidden)
    if ratio == 4:
        rnn_type = "lstm"
    elif ratio == 3:
        rnn_type = "gru"
    else:
        raise ValueError("%s.rnn.weight_ih_l0 has %d rows for hidden size %d; "
                         "expected 4*H (LSTM) or 3*H (GRU)" % (name, w_ih.shape[0], hidden))
    return rnn_type, int(hidden), len(layers), int(w_ih.shape[1])


def infer_arch(state_dict):
    actor = _mlp_linear_shapes(state_dict, "actor")
    critic = _mlp_linear_shapes(state_dict, "critic")
    rnn_a = _rnn_info(state_dict, "memory_a")
    rnn_c = _rnn_info(state_dict, "memory_c")
    if (rnn_a is None) != (rnn_c is None):
        raise ValueError("only one of memory_a / memory_c is present in state_dict")

    num_actions = actor[-1][0]
    if "std" in state_dict and tuple(state_dict["std"].shape) != (num_actions,):
        raise ValueError("std shape %s does not match actor output dim %d"
                         % (tuple(state_dict["std"].shape), num_actions))
    if critic[-1][0] != 1:
        raise ValueError("critic output dim is %d, expected 1" % critic[-1][0])

    arch = {
        "policy_class_name": "ActorCritic",
        "num_actor_obs": actor[0][1],
        "num_critic_obs": critic[0][1],
        "num_actions": num_actions,
        "actor_hidden_dims": [s[0] for s in actor[:-1]],
        "critic_hidden_dims": [s[0] for s in critic[:-1]],
        "rnn_type": None,
        "rnn_hidden_size": None,
        "rnn_num_layers": None,
    }
    if rnn_a is not None:
        if rnn_a[:3] != rnn_c[:3]:
            raise ValueError("actor/critic RNN configs differ: %s vs %s" % (rnn_a[:3], rnn_c[:3]))
        if actor[0][1] != rnn_a[1] or critic[0][1] != rnn_c[1]:
            raise ValueError("MLP input dim does not equal RNN hidden size %d" % rnn_a[1])
        arch.update({
            "policy_class_name": "ActorCriticRecurrent",
            "num_actor_obs": rnn_a[3],
            "num_critic_obs": rnn_c[3],
            "rnn_type": rnn_a[0],
            "rnn_hidden_size": rnn_a[1],
            "rnn_num_layers": rnn_a[2],
        })
    return arch


def load_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    if "model_state_dict" not in checkpoint:
        raise ValueError("%s has no 'model_state_dict' (keys: %s)" % (path, list(checkpoint.keys())))
    return checkpoint, checkpoint["model_state_dict"]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_arch_to_train_cfg(train_cfg, arch, activation="elu"):
    train_cfg.runner.policy_class_name = arch["policy_class_name"]
    train_cfg.policy.actor_hidden_dims = list(arch["actor_hidden_dims"])
    train_cfg.policy.critic_hidden_dims = list(arch["critic_hidden_dims"])
    train_cfg.policy.activation = activation
    if arch["rnn_type"] is not None:
        train_cfg.policy.rnn_type = arch["rnn_type"]
        train_cfg.policy.rnn_hidden_size = arch["rnn_hidden_size"]
        train_cfg.policy.rnn_num_layers = arch["rnn_num_layers"]
    return train_cfg
