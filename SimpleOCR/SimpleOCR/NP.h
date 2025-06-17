#pragma once
#include "Matrix.h"

namespace ocr
{
    namespace NP
    {
        Matrix VecMat(const Array<float>& b, const Matrix& A);

        Matrix MatVec(const Matrix& A, const Array<float>& b);

        Matrix MatMul(const Matrix& A, const Matrix& B);

        void GEMM(const Array<float>& a, const Matrix& B, Array<float>& c);

        void GEMM(const Matrix& A, const Matrix& B, Matrix& C);
    }
}
