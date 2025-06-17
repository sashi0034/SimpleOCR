#pragma once
#include "DatasetImage.h"

namespace ocr
{
    void LoadMnistImages(const std::string& file, DatasetImageList& images);
    void LoadMnistLabels(const std::string& file, Array<uint8_t>& labels);
}
