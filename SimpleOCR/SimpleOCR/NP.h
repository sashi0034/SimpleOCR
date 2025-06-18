#pragma once
#include "Matrix.h"

namespace ocr
{
    namespace NP
    {
        Array<float> Subtract(const Array<float>& a, const Array<float>& b);

        Array<float> Divide(const Array<float>& a, float b);

        Matrix VecMat(const Array<float>& b, const Matrix& A);

        Matrix MatVec(const Matrix& A, const Array<float>& b);

        Matrix MatMul(const Matrix& A, const Matrix& B);

        void GEMM(const Array<float>& a, const Matrix& B, Array<float>& c);

        void GEMM(const Matrix& A, const Matrix& B, Matrix& C);

        /// @テンソル積
        Matrix OuterProduct(const Array<float>& a, const Array<float>& b);

        /// @ドット積
        float DorProduct(const Array<float>& a, const Array<float>& b);

        /// @brief アダマール積
        Array<float> HadamardProduct(const Array<float>& a, const Array<float>& b);
    }
}
