"""Offline architecture gate for the paper's registered Isaac Gym checkpoints.

This module does not inspect checkpoint bytes. The archived registry record is
MLP ActorCritic [512, 256, 128]; upstream G1 LSTM defaults are intentionally
not accepted as an equivalent.
"""
PAPER_ARCHITECTURE = {
    "policy_class_name": "ActorCritic",
    "actor_hidden_dims": [512, 256, 128],
    "activation": "elu",
    "recurrent": False,
}


def validate_architecture(record):
    """Validate explicit registry metadata without deserializing a checkpoint."""
    if record.get("policy_class_name") != PAPER_ARCHITECTURE["policy_class_name"]:
        raise ValueError("paper checkpoint must declare ActorCritic, not %r" % record.get("policy_class_name"))
    dims = record.get("actor_hidden_dims")
    if isinstance(dims, str):
        import json
        dims = json.loads(dims)
    if list(dims) != PAPER_ARCHITECTURE["actor_hidden_dims"]:
        raise ValueError("paper actor must be MLP [512, 256, 128], got %r" % (dims,))
    if record.get("rnn_type"):
        raise ValueError("recurrent checkpoint is not the paper MLP: %r" % record["rnn_type"])
    return True
