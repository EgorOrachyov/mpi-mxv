import argparse
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
EXEC = "mpi_mxv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-folder", default="cmake-build-release")
    args = parser.parse_args()

    processes = [1, 2, 4, 8, 16, 32]
    iterations = [20]
    blocking = [1, 0]
    by_row = [1, 0]
    files_matrix = ["A_3000x3000.data"]
    files_vector = ["v_3000.data"]

    for f_m, f_v, i in zip(files_matrix, files_vector, iterations):
        for r in by_row:
            for b in blocking:
                for p in processes:
                    print(f"Exec it={i} procs={p} by_row={r} block={b} fs={f_m},{f_v}")
                    subprocess.check_call(
                        ["mpiexec", "-n", str(p), str(ROOT / args.exec_folder / EXEC),
                         str(ROOT / DATA / f_m), str(ROOT / DATA / f_v), str(i), str(b), str(r)])


if __name__ == '__main__':
    main()
