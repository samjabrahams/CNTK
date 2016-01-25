//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Utils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

void Utils::ExpandDotDotDot(std::wstring& featPath, const std::wstring& scpPath, std::wstring& scpDirCached)
{
    wstring delim = L"/\\";

    if (scpDirCached.empty())
    {
        scpDirCached = scpPath;
        wstring tail;
        auto pos = scpDirCached.find_last_of(delim);
        if (pos != wstring::npos)
        {
            tail = scpDirCached.substr(pos + 1);
            scpDirCached.resize(pos);
        }
        if (tail.empty()) // nothing was split off: no dir given, 'dir' contains the filename
            scpDirCached.swap(tail);
    }
    size_t pos = featPath.find(L"...");
    if (pos != featPath.npos)
        featPath = featPath.substr(0, pos) + scpDirCached + featPath.substr(pos + 3);
}

void Utils::CheckMinibatchSizes(const intargvector& numberOfuttsPerMinibatchForAllEpochs)
{
    for (int i = 0; i < numberOfuttsPerMinibatchForAllEpochs.size(); i++)
    {
        if (numberOfuttsPerMinibatchForAllEpochs[i] < 1)
        {
            LogicError("nbrUttsInEachRecurrentIter cannot be less than 1.");
        }
    }
}
} } }
