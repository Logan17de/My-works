#!/usr/bin/env python3
from __future__ import annotations

import argparse
from progress_utils import Progress
from universal_motion.sources import ARDYMotionSource

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--source",default="ardy",choices=["ardy"]); p.add_argument("--input",required=True); p.add_argument("--output",required=True); return p.parse_args()

def main():
    a=parse_args(); p=Progress("CANONICAL MOTION",3)
    p.step("Loading source Motion Adapter")
    adapter=ARDYMotionSource(); p.info("Motion source: ARDY Core27")
    p.step("Mapping source skeleton into canonical semantic motion")
    motion=adapter.load(a.input); p.info(f"Canonical joints={len(motion.joint_names)} | frames={motion.frames} | fps={motion.fps:g}"); p.info(f"Reference height={motion.reference_height:.4f}")
    p.step("Writing reusable canonical motion artifact")
    out=motion.save(a.output); p.ok(f"Canonical motion: {out} ({out.stat().st_size/1024**2:.2f} MiB)"); p.done("Motion Adapter + Skeleton Mapper complete")

if __name__=="__main__": main()
