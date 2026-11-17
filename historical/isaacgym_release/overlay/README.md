# Selected historical Isaac Gym overlay

The files beneath this directory are exact byte copies from a private source commit `8df0dfc8669620b5ee058dbe117fe92b653aafc6`, restricted to the candidate plane evaluation chain, with its root and legged_gym BSD notices. The real evaluator is `overlay/legged_gym/scripts/eval_batch.py`; `../eval_batch.py` is an offline stub, not the runtime.

Apply only to a separately owned, fresh checkout of `unitree_rl_gym`, never the existing upstream checkout. The recorded upstream commit is `276801e46c5d433564f24658bac64f254b7d2d4b` (recommended; other commits are allowed if the actual commit is logged). This is an unverified candidate overlay: static checks are not a simulator test. See `../../../docs/isaacgym_reproduction.md` for copy instructions, validation and limitations.
