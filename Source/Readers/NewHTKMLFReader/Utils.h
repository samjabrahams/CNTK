//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <vector>
#include <memory>

#include "Config.h"

#include "biggrowablevectors.h"
#include "utterancesourcemulti.h"
#include "minibatchiterator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class Utils
{
public:
    static void ExpandDotDotDot(std::wstring& featPath, const std::wstring& scpPath, std::wstring& scpDirCached);

    static void GetDataNamesFromConfig(
        const ConfigParameters& readerConfig,
        std::vector<std::wstring>& features,
        std::vector<std::wstring>& labels,
        std::vector<std::wstring>& hmms,
        std::vector<std::wstring>& lattices);

    static void CheckMinibatchSizes(const intargvector& numberOfuttsPerMinibatchForAllEpochs);
};
} } }
