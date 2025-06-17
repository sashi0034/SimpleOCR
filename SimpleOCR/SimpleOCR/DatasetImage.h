#pragma once
#include "TY/Array.h"
#include "TY/ImageView.h"

namespace ocr
{
    using namespace TY;

    struct DatasetImageProperty
    {
        Size size{};
    };

    class DatasetImage : public Array<uint8_t>
    {
    public:
        using Array::Array;

        ImageView imageView(const DatasetImageProperty& prop) const
        {
            return ImageView{
                (data()),
                prop.size,
                size_in_bytes(),
                DXGI_FORMAT_R8_UNORM,
            };
        }
    };

    struct DatasetImageList
    {
        DatasetImageProperty property{};
        Array<DatasetImage> images{};
    };
}
