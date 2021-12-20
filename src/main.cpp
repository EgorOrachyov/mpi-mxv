#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <numeric>

#include <mpi.h>

using size = std::uint64_t;

// Matrix in row-major format
struct Matrix {
    size M{}, N{};
    std::vector<double> d;
    mutable const double *dp = nullptr;
};

// Vector
using Vector = Matrix;

// Timer wrapper
struct Timer {
private:
    using clock = std::chrono::steady_clock;
    using ns = std::chrono::nanoseconds;
    clock::time_point t1, t2;
public:
    void begin() { t1 = t2 = clock::now(); }

    void end() { t2 = clock::now(); }

    double mark() {
        end();
        auto s = ms();
        t1 = t2;
        return s;
    }

    double ms() const { return static_cast<double>(std::chrono::duration_cast<ns>(t2 - t1).count()) / 1e6; }
};

// Load matrix (vector) from path
void load(const std::string &path, Matrix &matrix) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::exception("Failed open file");
    // Matrix properties
    file >> matrix.M >> matrix.N;
    size count = matrix.M * matrix.N;
    matrix.d.resize(count);
    // Load all values one by one (by row)
    for (size i = 0; i < count; i++) {
        file >> matrix.d[i];
    }
}

// Split M into set of segments
void decompose(size M, size count, std::vector<size> &slices) {
    size start = 0;
    for (size i = 0; i < count; i++) {
        size step = i == count - 1 ? M - start : M / count;
        slices.push_back(step);
        start += step;
    }
}

// Decompose matrix A into list of slices A[0:s,:], A[s:2s,:] ... where s is step
void decompose_by_row(const Matrix &A, size count, std::vector<Matrix> &slices) {
    size M = A.M;
    size start = 0;
    for (size i = 0; i < count; i++) {
        size step = i == count - 1 ? M - start : M / count;
        Matrix slice;
        slice.M = step;
        slice.N = A.N;
        slice.d.resize(slice.M * slice.N);
        std::copy(A.d.begin() + start * A.N, A.d.begin() + (start + step) * A.N, slice.d.begin());
        slices.push_back(std::move(slice));
        start += step;
    }
}

// Decompose matrix A into list of slices A[:,0:s], A[:,s:2s] ... where s is step
void decompose_by_col(const Matrix &A, size count, std::vector<Matrix> &slices) {
    size N = A.N;
    size start = 0;
    slices.resize(count);
    for (size j = 0; j < count; j++) {
        size step = j == count - 1 ? N - start : N / count;
        slices[j].M = A.M;
        slices[j].N = step;
        slices[j].d.resize(slices[j].M * slices[j].N);
        start += step;
    }

    for (size i = 0; i < A.M; i++) {
        start = 0;
        auto row = A.d.begin() + i * A.N;
        for (size j = 0; j < count; j++) {
            size step = j == count - 1 ? N - start : N / count;
            std::copy(row, row + step, slices[j].d.begin() + i * slices[j].N);
            row = row + step;
            start += step;
        }
    }
}

// Compare two matrices with threshold error
bool equals(const Matrix &A, const Matrix &B, double error = 0.00005) {
    if (A.M != B.M || A.N != B.N || A.d.size() != B.d.size())
        return false;
    for (size i = 0; i < A.d.size(); i++) {
        if (std::abs(A.d[i] - B.d[i]) > error)
            return false;
    }
    return true;
}

// MPI related globals
static int procRank;
static int procCount;
static const int MASTER = 0;
static std::ofstream out;
// Timing
std::vector<double> times;
std::vector<double> send;
std::vector<double> exec;
std::vector<double> recv;

// Naive implementation (also used to multiply blocks by each process)
void mxv_naive(Vector &w, const Matrix &A, const Vector &v) {
    assert(A.N == v.M);
    assert(v.N == 1);
    auto M = A.M, N = A.N;
    w.M = M;
    w.N = 1;
    w.d.resize(M);
    if (!A.dp) A.dp = A.d.data();
    if (!v.dp) v.dp = v.d.data();
    // For each result vector row
    for (size i = 0; i < M; i++) {
        double res = 0.0;
        // Compute dot product M[i,:] * A
        for (size j = 0; j < N; j++) {
            res += A.dp[i * N + j] * v.dp[j];
        }
        w.d[i] = res;
    }
}

// Blocking p2p mxv with by row matrix decomposition
void mxv_mpi_by_row_b(Vector &w, const Matrix &A, const Vector &v) {
    std::vector<size> slices;
    Matrix slice;
    Vector vec;
    const Matrix *A_block;
    const Vector *v_block;
    Timer t;

    t.begin();
    if (procRank == MASTER) {
        decompose(A.M, procCount, slices);
        size offset = slices[procRank] * A.N;
        for (int i = 1; i < procCount; i++) {
            auto m = slices[i];
            auto step = m * A.N;
            MPI_Send(&m, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
            MPI_Send(&A.N, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
            MPI_Send(A.d.data() + offset, static_cast<int>(step), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(v.d.data(), static_cast<int>(v.M), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            offset += step;
        }
        slice.M = slices[procRank];
        slice.N = A.N;
        slice.dp = A.d.data();
        A_block = &slice;
        v_block = &v;
    } else {
        MPI_Recv(&slice.M, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&slice.N, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        slice.d.resize(slice.M * slice.N);
        vec.M = slice.N;
        vec.N = 1;
        vec.d.resize(vec.M);
        MPI_Recv(slice.d.data(), static_cast<int>(slice.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(vec.d.data(), static_cast<int>(vec.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        A_block = &slice;
        v_block = &vec;
    }
    send.push_back(t.mark());

    mxv_naive(w, *A_block, *v_block);
    exec.push_back(t.mark());

    if (procRank == MASTER) {
        Vector w_final;
        w_final.M = A.M;
        w_final.N = 1;
        w_final.d.resize(w_final.M);
        std::copy(w.d.begin(), w.d.end(), w_final.d.begin());
        size offset = w.M;
        for (int i = 1; i < procCount; i++) {
            MPI_Recv(w_final.d.data() + offset, static_cast<int>(slices[i]), MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            offset += slices[i];
        }
        w = std::move(w_final);
    } else {
        MPI_Send(w.d.data(), static_cast<int>(slice.M), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
    }
    recv.push_back(t.mark());
}

// Non-blocking p2p mxv with by row matrix decomposition
void mxv_mpi_by_row_nb(Vector &w, const Matrix &A, const Vector &v) {
    std::vector<size> slices;
    Matrix slice;
    Vector vec;
    const Matrix *A_block;
    const Vector *v_block;
    Timer t;

    t.begin();
    if (procRank == MASTER) {
        decompose(A.M, procCount, slices);
        size offset = slices[procRank] * A.N;
        for (int i = 1; i < procCount; i++) {
            auto m = slices[i];
            auto step = m * A.N;
            MPI_Request request;
            MPI_Isend(&m, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(&A.N, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(A.d.data() + offset, static_cast<int>(step), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(v.d.data(), static_cast<int>(v.M), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
            offset += step;
        }
        slice.M = slices[procRank];
        slice.N = A.N;
        slice.dp = A.d.data();
        A_block = &slice;
        v_block = &v;
    } else {
        std::vector<MPI_Request> requests(2);
        MPI_Irecv(&slice.M, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&slice.N, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests.data(), MPI_STATUSES_IGNORE);
        slice.d.resize(slice.M * slice.N);
        vec.M = slice.N;
        vec.N = 1;
        vec.d.resize(vec.M);
        MPI_Irecv(slice.d.data(), static_cast<int>(slice.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD,
                  &requests[0]);
        MPI_Irecv(vec.d.data(), static_cast<int>(vec.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests.data(), MPI_STATUSES_IGNORE);
        A_block = &slice;
        v_block = &vec;
    }
    send.push_back(t.mark());

    mxv_naive(w, *A_block, *v_block);
    exec.push_back(t.mark());

    if (procRank == MASTER) {
        Vector w_final;
        w_final.M = A.M;
        w_final.N = 1;
        w_final.d.resize(w_final.M);
        std::vector<MPI_Request> requests(procCount - 1);
        size offset = w.M;
        for (int i = 1; i < procCount; i++) {
            MPI_Irecv(w_final.d.data() + offset, static_cast<int>(slices[i]), MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                      &requests[i - 1]);
            offset += slices[i];
        }
        std::copy(w.d.begin(), w.d.end(), w_final.d.begin());
        MPI_Waitall(procCount - 1, requests.data(), MPI_STATUSES_IGNORE);
        w = std::move(w_final);
    } else {
        MPI_Request request;
        MPI_Isend(w.d.data(), static_cast<int>(slice.M), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    recv.push_back(t.mark());
}

// Blocking p2p mxv with by column matrix decomposition
void mxv_mpi_by_col_b(Vector &w, const Matrix &A, const Vector &v) {
    std::vector<Matrix> m_slices;
    std::vector<Vector> v_slices;
    Matrix slice;
    Vector vec;
    const Matrix *A_block;
    const Vector *v_block;
    Timer t;

    t.begin();
    if (procRank == MASTER) {
        decompose_by_col(A, procCount, m_slices);
        decompose_by_row(v, procCount, v_slices);
        for (int i = 1; i < procCount; i++) {
            auto &m_d = m_slices[i];
            auto &v_d = v_slices[i];
            MPI_Send(&m_d.M, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
            MPI_Send(&m_d.N, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
            MPI_Send(m_d.d.data(), static_cast<int>(m_d.d.size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(v_d.d.data(), static_cast<int>(v_d.d.size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        A_block = &m_slices[procRank];
        v_block = &v_slices[procRank];
    } else {
        MPI_Recv(&slice.M, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&slice.N, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        slice.d.resize(slice.M * slice.N);
        vec.M = slice.N;
        vec.N = 1;
        vec.d.resize(vec.M);
        MPI_Recv(slice.d.data(), static_cast<int>(slice.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(vec.d.data(), static_cast<int>(vec.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        A_block = &slice;
        v_block = &vec;
    }
    send.push_back(t.mark());

    mxv_naive(w, *A_block, *v_block);
    exec.push_back(t.mark());

    if (procRank == MASTER) {
        std::vector<double> tmp(w.M);
        for (int i = 1; i < procCount; i++) {
            MPI_Recv(tmp.data(), static_cast<int>(tmp.size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size k = 0; k < tmp.size(); k++)
                w.d[k] += tmp[k];
        }
    } else {
        MPI_Send(w.d.data(), static_cast<int>(w.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
    }
    recv.push_back(t.mark());
}

// Non-blocking p2p mxv with by column matrix decomposition
void mxv_mpi_by_col_nb(Vector &w, const Matrix &A, const Vector &v) {
    std::vector<Matrix> m_slices;
    std::vector<Vector> v_slices;
    Matrix slice;
    Vector vec;
    const Matrix *A_block;
    const Vector *v_block;
    Timer t;

    t.begin();
    if (procRank == MASTER) {
        decompose_by_col(A, procCount, m_slices);
        decompose_by_row(v, procCount, v_slices);
        for (int i = 1; i < procCount; i++) {
            auto &m_d = m_slices[i];
            auto &v_d = v_slices[i];
            MPI_Request request;
            MPI_Isend(&m_d.M, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(&m_d.N, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(m_d.d.data(), static_cast<int>(m_d.d.size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
            MPI_Isend(v_d.d.data(), static_cast<int>(v_d.d.size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
        }
        A_block = &m_slices[procRank];
        v_block = &v_slices[procRank];
    } else {
        std::vector<MPI_Request> requests(2);
        MPI_Irecv(&slice.M, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&slice.N, 1, MPI_UINT64_T, MASTER, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests.data(), MPI_STATUSES_IGNORE);
        slice.d.resize(slice.M * slice.N);
        vec.M = slice.N;
        vec.N = 1;
        vec.d.resize(vec.M);
        MPI_Irecv(slice.d.data(), static_cast<int>(slice.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD,
                  &requests[0]);
        MPI_Irecv(vec.d.data(), static_cast<int>(vec.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests.data(), MPI_STATUSES_IGNORE);
        A_block = &slice;
        v_block = &vec;
    }
    send.push_back(t.mark());

    mxv_naive(w, *A_block, *v_block);
    exec.push_back(t.mark());

    if (procRank == MASTER) {
        std::vector<int> received(procCount - 1, 0);
        std::vector<MPI_Request> requests(procCount - 1);
        std::vector<std::vector<double>> tmps(procCount - 1, std::vector<double>(w.M));
        for (int i = 1; i < procCount; i++) {
            MPI_Irecv(tmps[i - 1].data(), static_cast<int>(tmps[i - 1].size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                      &requests[i - 1]);
        }
        int merged = 0;
        while (merged < procCount - 1) {
            for (int i = 0; i < procCount - 1; i++) {
                if (received[i])
                    continue;
                MPI_Test(&requests[i], &received[i], MPI_STATUS_IGNORE);
                if (received[i]) {
                    for (size k = 0; k < tmps[i].size(); k++)
                        w.d[k] += tmps[i][k];
                    merged += 1;
                }
            }
        }
    } else {
        MPI_Request request;
        MPI_Isend(w.d.data(), static_cast<int>(w.d.size()), MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    recv.push_back(t.mark());
}

void stats(const char *text, const std::vector<double> &t) {
    auto iterations = t.size();
    double average = std::reduce(t.begin(), t.end(), 0.0) / static_cast<double>(iterations);
    double min_time = *std::min_element(t.begin(), t.end());
    double max_time = *std::max_element(t.begin(), t.end());
    double sd2 = std::transform_reduce(t.begin(), t.end(), 0.0, std::plus<>(),
                                       [=](auto x) { return (x - average) * (x - average); })
                 / static_cast<double>(iterations - 1);
    double sd = std::sqrt(sd2);

    out << text << " "
        << average << " "
        << min_time << " "
        << max_time << " "
        << sd << " (in ms)" << std::endl;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    assert(argc >= 6);

    out.open(std::string{"mpi_mxv_"} + std::to_string(procRank) + ".txt", std::ios_base::app);

    Matrix A;
    Vector v;
    Vector w;
    Vector w_ref;
    std::string A_path = argv[1];
    std::string v_path = argv[2];
    int iterations = std::atoi(argv[3]);
    bool blocking = std::atoi(argv[4]);
    bool by_row = std::atoi(argv[5]);

    out << "Exec mxv: "
        << "iters: " << iterations << " "
        << "blocking: " << blocking << " "
        << "by row: " << by_row << " "
        << "Pr ct: " << procCount << " "
        << "Pr rk: " << procRank << " ";

    if (procRank == MASTER) {
        Timer t;
        t.begin();
        load(A_path, A);
        load(v_path, v);
        t.end();
        out << "Load data " << t.ms() << " ms\n";
    }

    // Reference evaluation
    if (procRank == MASTER)
        mxv_naive(w_ref, A, v);

    for (int i = 0; i < iterations; i++) {
        Timer t;

        t.begin();
        if (procCount > 1) {
            // MPI based mxv evaluation
            if (by_row) {
                if (blocking)
                    mxv_mpi_by_row_b(w, A, v);
                else
                    mxv_mpi_by_row_nb(w, A, v);
            } else {
                if (blocking)
                    mxv_mpi_by_col_b(w, A, v);
                else
                    mxv_mpi_by_col_nb(w, A, v);
            }
        } else {
            // Naive single thread reference implementation
            mxv_naive(w, A, v);
        }
        times.push_back(t.mark());

        // If in multi-threaded mode - check result
        if (procRank == MASTER)
            out << "eq: i" << i << "=" << equals(w_ref, w) << ", ";
    }

    if (procRank == MASTER) {
        out << "\n";
        stats("all", times);
        if (procCount > 1) {
            stats("send", send);
            stats("exec", exec);
            stats("recv", recv);
        }
    }

    MPI_Finalize();
    return 0;
}