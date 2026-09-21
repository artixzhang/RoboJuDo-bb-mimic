"""Run the same G1 basketball student in MuJoCo or on G1 EDU hardware."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robojudo.deploy.g1_shoot import load_config, run_real, run_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim", "real", "check"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "robojudo/deploy/g1_shoot.yaml")
    parser.add_argument("--onnx", help="Override the ONNX path in the deployment YAML")
    parser.add_argument("--net", help="Override the real robot network interface")
    parser.add_argument("--real-log", type=Path, help="Write real q/target/torque telemetry to CSV")
    args = parser.parse_args()
    cfg = load_config(args.config, args.onnx)
    if args.net:
        cfg["real"]["network_interface"] = args.net
    if args.real_log:
        cfg["real"]["log_path"] = str(args.real_log)
    if args.mode == "real":
        run_real(cfg)
    else:
        run_sim(cfg, check_steps=cfg["phase_frames"] + 40 if args.mode == "check" else 0)


if __name__ == "__main__":
    main()
