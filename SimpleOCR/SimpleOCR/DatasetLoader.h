#pragma once
#include "DatasetImage.h"

namespace ocr
{
    DatasetImageList LoadMnistImages(const std::string& file);
    Array<uint8_t> LoadMnistLabels(const std::string& file);
}
