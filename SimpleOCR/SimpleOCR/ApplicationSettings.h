#pragma once

namespace ocr
{
    struct ApplicationSettings
    {
        bool useGpu = true;
    };

    inline ApplicationSettings g_applicationSettings{};
}
