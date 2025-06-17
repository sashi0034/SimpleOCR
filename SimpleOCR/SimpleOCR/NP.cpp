#include "pch.h"
#include "NP.h"

namespace ocr
{
    Matrix NP::VecMat(const Array<float>& b, const Matrix& A)
    {
        if (A.rows() != b.size())
        {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        Matrix result(1, A.cols());
        for (int j = 0; j < A.cols(); ++j)
        {
            result[0][j] = 0;
            for (int i = 0; i < A.rows(); ++i)
            {
                result[0][j] += b[i] * A[i][j];
            }
        }

        return result;
    }

    Matrix NP::MatVec(const Matrix& A, const Array<float>& b)
    {
        if (A.cols() != b.size())
        {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        Matrix result(A.rows(), 1);
        for (int i = 0; i < A.rows(); ++i)
        {
            result[i][0] = 0;
            for (int j = 0; j < A.cols(); ++j)
            {
                result[i][0] += A[i][j] * b[j];
            }
        }

        return result;
    }

    Matrix NP::MatMul(const Matrix& A, const Matrix& B)
    {
        if (A.cols() != B.rows())
        {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix result(A.rows(), B.cols());
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                result[i][j] = 0;
                for (int k = 0; k < A.cols(); ++k)
                {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    void NP::GEMM(const Array<float>& a, const Matrix& B, Array<float>& c)
    {
        if (B.rows() != a.size() || B.cols() != c.size())
        {
            throw std::invalid_argument("Matrix and vector dimensions do not match for GEMM operation.");
        }

        for (int j = 0; j < B.cols(); ++j)
        {
            c[j] = 0;
            for (int i = 0; i < B.rows(); ++i)
            {
                c[j] += a[i] * B[i][j];
            }
        }
    }

    void NP::GEMM(const Matrix& A, const Matrix& B, Matrix& C)
    {
        if (A.cols() != B.rows() || A.rows() != C.rows() || B.cols() != C.cols())
        {
            throw std::invalid_argument("Matrix dimensions do not match for GEMM operation.");
        }

        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                C[i][j] = 0;
                for (int k = 0; k < A.cols(); ++k)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}
