﻿#pragma once
#include "TY/Array.h"
#include "TY/Vector2D.h"

using namespace TY;

namespace ocr
{
    struct Matrix
    {
        Matrix() = default;

        Matrix(int rows, int cols);

        int rows() const
        {
            return m_rows;
        }

        int cols() const
        {
            return m_cols;
        }

        Size colsRows() const
        {
            return Size{m_cols, m_rows};
        }

        const Array<float>& data() const
        {
            return m_data;
        }

        Array<float>& data()
        {
            return m_data;
        }

        float* operator[](int index)
        {
            return &m_data[index * m_cols];
        }

        const float* operator[](int index) const
        {
            return &m_data[index * m_cols];
        }

        Matrix transposed() const;

        static Matrix RowMajor(Array<float> vector);

        static Matrix ColumnMajor(Array<float> vector);

    private:
        /// @brief 行数
        int m_rows;

        /// @brief 列数
        int m_cols;

        Array<float> m_data;
    };
}
