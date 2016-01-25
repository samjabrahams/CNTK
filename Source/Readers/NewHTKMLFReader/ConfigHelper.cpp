//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "ConfigHelper.h"
#include <Config.h>
#include <DataReader.h>
#include <regex>
#include "Utils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::pair<size_t, size_t> ConfigHelper::GetContextWindow(const ConfigParameters& config)
{
    size_t left = 0, right = 0;
    intargvector contextWindow = config(L"contextWindow", ConfigParameters::Array(intargvector(vector<int>{1})));

    if (contextWindow.size() == 1) // symmetric
    {
        size_t windowFrames = contextWindow[0];
        if (windowFrames % 2 == 0)
        {
            InvalidArgument("augmentationextent: neighbor expansion of input features to %lu not symmetrical", windowFrames);
        }

        // extend each side by this
        size_t context = windowFrames / 2;
        left = context;
        right = context;
    }
    else if (contextWindow.size() == 2)
    {
        // left context, right context
        left = contextWindow[0];
        right = contextWindow[1];
    }
    else
    {
        InvalidArgument("contextFrames must have 1 or 2 values specified, found %lu", contextWindow.size());
    }

    return std::make_pair(left, right);
}

void ConfigHelper::CheckFeatureType(const ConfigParameters& config)
{
    std::wstring type = config(L"type", L"real");
    if (_wcsicmp(type.c_str(), L"real"))
    {
        InvalidArgument("Feature type must be 'real'.");
    }
}

void ConfigHelper::CheckLabelType(const ConfigParameters& config)
{
    std::wstring type;
    if (config.Exists(L"labelType"))
    {
        // let's deprecate this eventually and just use "type"...
        type = static_cast<const std::wstring&>(config(L"labelType"));
    }
    else
    {
        // outputs should default to category
        type = static_cast<const std::wstring&>(config(L"type", L"category"));
    }

    if (_wcsicmp(type.c_str(), L"category"))
    {
        InvalidArgument("label type must be 'category'");
    }
}

// GetFileConfigNames - determine the names of the features and labels sections in the config file
// features - [in,out] a vector of feature name strings
// labels - [in,out] a vector of label name strings
void ConfigHelper::GetDataNamesFromConfig(
    const ConfigParameters& readerConfig,
    std::vector<std::wstring>& features,
    std::vector<std::wstring>& labels,
    std::vector<std::wstring>& hmms,
    std::vector<std::wstring>& lattices)
{
    for (const auto& id : readerConfig.GetMemberIds())
    {
        if (!readerConfig.CanBeConfigRecord(id))
            continue;
        const ConfigParameters& temp = readerConfig(id);
        // see if we have a config parameters that contains a "file" element, it's a sub key, use it
        if (temp.ExistsCurrent(L"scpFile"))
        {
            features.push_back(id);
        }
        else if (temp.ExistsCurrent(L"mlfFile") || temp.ExistsCurrent(L"mlfFileList"))
        {
            labels.push_back(id);
        }
        else if (temp.ExistsCurrent(L"phoneFile"))
        {
            hmms.push_back(id);
        }
        else if (temp.ExistsCurrent(L"denlatTocFile"))
        {
            lattices.push_back(id);
        }
    }
}

size_t ConfigHelper::GetFeatureDimension(const ConfigParameters& config)
{
    if (config.Exists(L"dim"))
    {
        return config(L"dim");
    }

    InvalidArgument("features must specify dim");
}

size_t ConfigHelper::GetLabelDimension(const ConfigParameters& config)
{
    if (config.Exists(L"labelDim"))
    {
        return config(L"labelDim");
    }

    if (config.Exists(L"dim"))
    {
        return config(L"dim");
    }

    InvalidArgument("labels must specify dim or labelDim");
}

std::vector<std::wstring> ConfigHelper::GetMlfPaths(const ConfigParameters& config)
{
    std::vector<std::wstring> result;
    if (config.ExistsCurrent(L"mlfFile"))
    {
        result.push_back(config(L"mlfFile"));
    }
    else
    {
        if (!config.ExistsCurrent(L"mlfFileList"))
        {
            InvalidArgument("Either mlfFile or mlfFileList must exist in NewHTKMLFReader");
        }

        wstring list = config(L"mlfFileList");
        for (msra::files::textreader r(list); r;)
        {
            result.push_back(r.wgetline());
        }
    }

    return result;
}

size_t ConfigHelper::GetRandomizationWindow(const ConfigParameters& config)
{
    size_t result = randomizeAuto;

    if (config.Exists(L"randomize"))
    {
        wstring randomizeString = config.CanBeString(L"randomize") ? config(L"randomize") : wstring();
        if (!_wcsicmp(randomizeString.c_str(), L"none"))
        {
            result = randomizeNone;
        }
        else if (!_wcsicmp(randomizeString.c_str(), L"auto"))
        {
            result = randomizeAuto;
        }
        else
        {
            result = config(L"randomize");
        }
    }

    return result;
}

std::wstring ConfigHelper::GetRandomizer(const ConfigParameters& config)
{
    // get the read method, defaults to "blockRandomize" other option is "rollingWindow"
    std::wstring randomizer(config(L"readMethod", L"blockRandomize"));

    if (randomizer == L"blockRandomize" && ConfigHelper::GetRandomizationWindow(config) == randomizeNone)
    {
        InvalidArgument("'randomize' cannot be 'none' when 'readMethod' is 'blockRandomize'.");
    }

    return randomizer;
}

std::vector<std::wstring> ConfigHelper::GetFeaturePaths(const ConfigParameters& config)
{
    std::wstring scriptPath = config(L"scpFile");
    std::wstring rootPath = config(L"prefixPathInSCP", L"");

    vector<wstring> filelist;
    fprintf(stderr, "reading script file %ls ...", scriptPath.c_str());

    size_t n = 0;
    for (msra::files::textreader reader(scriptPath); reader;)
    {
        filelist.push_back(reader.wgetline());
        n++;
    }

    fprintf(stderr, " %d entries\n", static_cast<int>(n));

    // post processing file list :
    //  - if users specified PrefixPath, add the prefix to each of path in filelist
    //  - else do the dotdotdot expansion if necessary
    if (!rootPath.empty()) // use has specified a path prefix for this  feature
    {
        // first make slash consistent (sorry for Linux users:this is not necessary for you)
        std::replace(rootPath.begin(), rootPath.end(), L'\\', L'/');

        // second, remove trailing slash if there is any
        std::wregex trailer(L"/+$");
        rootPath = std::regex_replace(rootPath, trailer, wstring());

        // third, join the rootPath with each entry in filelist
        if (!rootPath.empty())
        {
            for (wstring& path : filelist)
            {
                if (path.find_first_of(L'=') != wstring::npos)
                {
                    std::vector<std::wstring> strarr = msra::strfun::split(path, L"=");
#ifdef WIN32
                    replace(strarr[1].begin(), strarr[1].end(), L'\\', L'/');
#endif
                    path = strarr[0] + L"=" + rootPath + L"/" + strarr[1];
                }
                else
                {
#ifdef WIN32
                    replace(path.begin(), path.end(), L'\\', L'/');
#endif
                    path = rootPath + L"/" + path;
                }
            }
        }
    }
    else
    {
        /*
                do "..." expansion if SCP uses relative path names
                "..." in the SCP means full path is the same as the SCP file
                for example, if scp file is "//aaa/bbb/ccc/ddd.scp"
                and contains entry like
                .../file1.feat
                .../file2.feat
                etc.
                the features will be read from
                //aaa/bbb/ccc/file1.feat
                //aaa/bbb/ccc/file2.feat
                etc.
                This works well if you store the scp file with the features but
                do not want different scp files everytime you move or create new features
                */
        std::wstring scpDirCached;
        for (auto& entry : filelist)
        {
            Utils::ExpandDotDotDot(entry, scriptPath, scpDirCached);
        }
    }

    return filelist;
}
} } }
