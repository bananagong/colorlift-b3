"""Path-only harness around the frozen round1 trainer and dataset implementation."""
import argparse
import os
from .spec import RUN, INITIAL, SEEDS, training_context, atomic, now


def main(group,seed):
    assert group in ("B2","B3") and seed in SEEDS
    assert os.environ["CUDA_VISIBLE_DEVICES"] in ("1","2")
    training_context()
    from ..broad_act import train as original_train, data as original_data
    # The approved data and algorithm are unchanged. Only artifact destinations
    # and the identical frozen initialization copy are redirected.
    original_train.RUN = RUN; original_train.INITIAL = INITIAL
    original_train.freeze = training_context; original_data.RUN = RUN
    atomic(RUN / "models" / f"{group}_s{seed}" / "run_boundary.json",{
        "started_at_UTC":now(),"group":group,"training_seed":seed,
        "physical_GPU":int(os.environ["CUDA_VISIBLE_DEVICES"]),"fresh_confirmation_read":False})
    original_train.train(f"{group}_s{seed}",seed,"balanced",.1 if group=="B3" else 0.,0.,False)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--group",required=True,choices=("B2","B3"));p.add_argument("--seed",type=int,required=True)
    a=p.parse_args();main(a.group,a.seed)
