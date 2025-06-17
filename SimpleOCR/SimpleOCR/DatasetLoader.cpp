#include "pch.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

#include "DatasetImage.h"

namespace
{
    uint32_t readBigEndianUint32(std::ifstream& ifs)
    {
        uint32_t value;
        ifs.read(reinterpret_cast<char*>(&value), 4);
        return ((value & 0xFF) << 24) |
            ((value & 0xFF00) << 8) |
            ((value & 0xFF0000) >> 8) |
            ((value & 0xFF000000) >> 24);
    }
}

namespace ocr
{
    void LoadMnistImages(const std::string& file, DatasetImageList& images)
    {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs) { throw std::runtime_error("Can't open file!"); }

        const int magic = readBigEndianUint32(ifs);
        const int num_images = readBigEndianUint32(ifs);
        const int rows = readBigEndianUint32(ifs);
        const int cols = readBigEndianUint32(ifs);

        images.property.size = {rows, cols};

        images.images.resize(num_images, DatasetImage(rows * cols));

        for (uint32_t i = 0; i < num_images; ++i)
        {
            ifs.read(reinterpret_cast<char*>(images.images[i].data()), rows * cols);
        }
    }

    void LoadMnistLabels(const std::string& file, Array<uint8_t>& labels)
    {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs) { throw std::runtime_error("Can't open file!"); }

        const int magic = readBigEndianUint32(ifs);
        const int num_labels = readBigEndianUint32(ifs);

        labels.resize(num_labels);
        ifs.read(reinterpret_cast<char*>(labels.data()), num_labels);
    }
}
