//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <utility>
#include <string>
#include <vector>
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ConfigHelper
{
public:
    static std::pair<size_t, size_t> GetContextWindow(const ConfigParameters& config);
    static size_t GetFeatureDimension(const ConfigParameters& config);
    static size_t GetLabelDimension(const ConfigParameters& config);

    static void CheckFeatureType(const ConfigParameters& config);
    static void CheckLabelType(const ConfigParameters& config);

    static void GetDataNamesFromConfig(
        const ConfigParameters& readerConfig,
        std::vector<std::wstring>& features,
        std::vector<std::wstring>& labels,
        std::vector<std::wstring>& hmms,
        std::vector<std::wstring>& lattices);

    static std::vector<std::wstring> GetMlfPaths(const ConfigParameters& config);
    static std::vector<std::wstring> GetFeaturePaths(const ConfigParameters& config);

    static size_t GetRandomizationWindow(const ConfigParameters& config);
    static std::wstring GetRandomizer(const ConfigParameters& config);
};
} } }
