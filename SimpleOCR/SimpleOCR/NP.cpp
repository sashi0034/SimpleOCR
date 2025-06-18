#include "pch.h"
#include "NP.h"

namespace ocr
{
    Array<float> NP::Subtract(const Array<float>& a, const Array<float>& b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Array sizes do not match for subtraction.");
        }

        Array<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = a[i] - b[i];
        }

        return result;
    }

    Array<float> NP::Divide(const Array<float>& a, float b)
    {
        if (b == 0)
        {
            throw std::invalid_argument("Division by zero is not allowed.");
        }

        Array<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = a[i] / b;
        }

        return result;
    }

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
                for (int k = 0; k < A.cols(); ++k)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    Matrix NP::OuterProduct(const Array<float>& a, const Array<float>& b)
    {
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            for (size_t j = 0; j < b.size(); ++j)
            {
                result[i][j] = a[i] * b[j];
            }
        }

        return result;
    }

    float NP::DorProduct(const Array<float>& a, const Array<float>& b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Array sizes do not match for dot product.");
        }

        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            result += a[i] * b[i];
        }

        return result;
    }

    Array<float> NP::HadamardProduct(const Array<float>& a, const Array<float>& b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Array sizes do not match for Hadamard product.");
        }

        Array<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = a[i] * b[i];
        }

        return result;
    }
}
