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

    Matrix Matrix::transposed() const
    {
        Matrix result(m_cols, m_rows);
        for (int i = 0; i < m_rows; ++i)
        {
            for (int j = 0; j < m_cols; ++j)
            {
                result[j][i] = m_data[i * m_cols + j];
            }
        }

        return result;
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
