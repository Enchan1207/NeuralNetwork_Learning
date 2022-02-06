#
# NeuralNetwork_Learning
#
import sys
from typing import List


def main(args: List[str]) -> int:

    return 0


if __name__ == "__main__":
    result = 0
    try:
        result = main(sys.argv) or 0
    except KeyboardInterrupt:
        print("Ctrl+C")
        exit(result)
