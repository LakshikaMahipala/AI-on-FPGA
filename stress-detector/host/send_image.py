"""Send an image to the FPGA over a serial port and print the prediction.

    pip install pyserial pillow
    python host/send_image.py --port COM5 face.jpg        # Linux: /dev/ttyUSB0
"""
import argparse

import numpy as np
import serial
from PIL import Image

LABELS = {0: "not stressed", 1: "stressed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    img = np.asarray(Image.open(args.image).convert("L").resize((48, 48)), dtype=np.uint8)
    with serial.Serial(args.port, args.baud, timeout=5) as port:
        port.write(img.tobytes())          # 2,304 bytes, row-major
        reply = port.read(1)
    if not reply:
        raise SystemExit("no reply from the FPGA (check port, baud rate and reset)")
    print("FPGA prediction:", LABELS.get(reply[0], f"unknown ({reply[0]})"))


if __name__ == "__main__":
    main()
