"""Install the checkout's Rust toolchain once, before probe fanout begins."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    checkout = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=checkout,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    active = subprocess.run(
        ["rustup", "show", "active-toolchain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]
    destination.write_text(active + "\n")


if __name__ == "__main__":
    main()
