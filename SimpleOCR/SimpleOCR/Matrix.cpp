#include "pch.h"
#include "Matrix.h"

namespace ocr
{
    Matrix::Matrix(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }

        this->m_rows = rows;
        this->m_cols = cols;
        m_data.resize(rows * cols);
    }

    Matrix Matrix::RowMajor(Array<float> vector)
    {
        Matrix result{};
        result.m_rows = 1;
        result.m_cols = vector.size();
        result.m_data = std::move(vector);
        return result;
    }

    Matrix Matrix::ColumnMajor(Array<float> vector)
    {
        Matrix result{};
        result.m_rows = vector.size();
        result.m_cols = 1;
        result.m_data = std::move(vector);
        return result;
    }
}
