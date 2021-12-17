#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>

#include <mpi.h>

using std::size_t;

// Matrix in row-major format
struct Matrix {
    std::size_t M{}, N{};
    std::vector<double> d;

    void print() {
        size_t row = 0;
        std::cout << "M=" << M << " N=" << N << "\n";
        std::for_each(d.begin(), d.end(), [&](double x) {
            row++;
            std::cout << x << " ";
            if (row == N) {
                row = 0;
                std::cout << "\n";
            }
        });
    }
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

    double sec() { return static_cast<double>(std::chrono::duration_cast<ns>(t2 - t1).count()) / 1e9; }
};

// Load matrix (vector) from path
void load(const std::string &path, Matrix &matrix) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::exception("Failed open file");
    // Matrix properties
    file >> matrix.M >> matrix.N;
    size_t count = matrix.M * matrix.N;
    matrix.d.resize(count);
    // Load all values one by one (by row)
    for (size_t i = 0; i < count; i++) {
        file >> matrix.d[i];
    }
}

// Decompose matrix A into list of slices A[0:s,:], A[s:2s,:] ... where s is step
void decompose_by_row(const Matrix &A, size_t count, std::vector<Matrix> &slices) {
    size_t M = A.M;
    size_t start = 0;
    for (size_t i = 0; i < count; i++) {
        size_t step = i == count - 1 ? M - start : M / count;
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
void decompose_by_col(const Matrix &A, size_t count, std::vector<Matrix> &slices) {
    size_t N = A.N;
    size_t start = 0;
    slices.resize(count);
    for (size_t j = 0; j < count; j++) {
        size_t step = j == count - 1 ? N - start : N / count;
        slices[j].M = A.M;
        slices[j].N = step;
        slices[j].d.resize(slices[j].M * slices[j].N);
        start += step;
    }

    for (size_t i = 0; i < A.M; i++) {
        start = 0;
        auto row = A.d.begin() + i * A.N;
        for (size_t j = 0; j < count; j++) {
            size_t step = j == count - 1 ? N - start : N / count;
            std::copy(row, row + step, slices[j].d.begin() + i * slices[j].N);
            row = row + step;
            start += step;
        }
    }
}

// Merge slices A[0:s,:], A[s:2s,:] into a matrix A
void merge_rows(const std::vector<Matrix> &slices, Matrix &A) {
    size_t M = 0;
    size_t N = slices.front().N;
    std::for_each(slices.begin(), slices.end(), [&](const Matrix &slice) { M += slice.M; });
    A.M = M;
    A.N = N;
    A.d.resize(M * N);
    size_t start = 0;
    std::for_each(slices.begin(), slices.end(), [&](const Matrix &slice) {
        std::copy(slice.d.begin(), slice.d.end(), A.d.begin() + start * N);
        start += slice.M;
    });
}

// Merge slices A[:,0:s], A[:,s:2s] into a matrix A
void merge_cols(const std::vector<Matrix> &slices, Matrix &A) {
    size_t N = 0;
    size_t M = slices.front().M;
    std::for_each(slices.begin(), slices.end(), [&](const Matrix &slice) { N += slice.N; });
    A.M = M;
    A.N = N;
    A.d.resize(M * N);
    for (size_t i = 0; i < A.M; i++) {
        size_t start = 0;
        auto row = A.d.begin() + i * A.N;
        for (size_t j = 0; j < slices.size(); j++) {
            size_t step = slices[j].N;
            std::copy(slices[j].d.begin() + i * slices[j].N, slices[j].d.begin() + i * slices[j].N + step, row);
            row = row + step;
            start += step;
        }
    }
}

// Compare two matrices with threshold error
bool equals(const Matrix &A, const Matrix &B, double error = 0.00005) {
    if (A.M != B.M || A.N != B.N || A.d.size() != B.d.size())
        return false;
    for (size_t i = 0; i < A.d.size(); i++) {
        if (std::abs(A.d[i] - B.d[i]) > error)
            return false;
    }
    return true;
}

// Naive implementation
void mxv_naive(Vector &w, const Matrix &A, const Vector &v) {
    assert(A.N == v.M);
    assert(v.N == 1);
    auto M = A.M, N = A.N;
    w.M = M;
    w.N = 1;
    w.d.resize(M);
    // For each result vector row
    for (size_t i = 0; i < M; i++) {
        double res = 0.0;
        // Compute dot product M[i,:] * A
        for (size_t j = 0; j < N; j++) {
            res += A.d[i * N + j] * v.d[j];
        }
        w.d[i] = res;
    }
}

void mpi_by_row(Vector &w, const Matrix &A, const Vector &v, bool nonblocking) {

}

int main(int, const char *const *) {
    int argc;
    char **argv;

//    Matrix A;
//    Vector v;
//    Vector w_ref;
//    Vector w_mpi;

//    load("../data/A_3000x3000.data", A);
//    load("../data/v_3000.data", v);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        MPI_Send("hello", 6, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    } else {
        char buffer[100];
        MPI_Recv(buffer, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        std::cout << buffer;
    }

    MPI_Finalize();

    return 0;
}