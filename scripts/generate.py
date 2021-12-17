import argparse

import numpy


def generate(m, n, dist=1.0):
    return numpy.random.uniform(-dist, dist, size=(m * n))


def save(path, m, n, matrix):
    with open(path, "w") as file:
        file.write(f"{m} {n}\n")
        row = 0
        for i, v in enumerate(matrix):
            row += 1
            file.write(f"{v} ")
            if row == n:
                row = 0
                file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default=3000)
    parser.add_argument("--n", default=1)
    parser.add_argument("--dist", default=50.0)
    parser.add_argument("--path", default="v_3000.data")
    args = parser.parse_args()

    data = generate(args.m, args.n, args.dist)
    save(args.path, args.m, args.n, data)


if __name__ == '__main__':
    main()
