import yaml

from quant_rotor import run_job


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run_job(**cfg)


if __name__ == "__main__":
    main()
