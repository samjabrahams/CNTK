//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"

#include "htkfeatio.h" // for reading HTK features
#ifdef _WIN32
#include "latticearchive.h" // for reading HTK phoneme lattices (MMI training)
#endif
#include "simplesenonehmm.h" // for MMI scoring
#include "msra_mgram.h"      // for unigram scores of ground-truth path in sequence training

#include "utterancesourcemulti.h" // minibatch sources
#include "chunkevalsource.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "NewHTKMLFReaderShim.h"
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void DATAREADER_API GetReader(IDataReader<ElemType>** preader)
{
    *preader = new NewHTKMLFReaderShim<ElemType>();
}

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    GetReader(preader);
}
extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    GetReader(preader);
}
#ifdef _WIN32
// Utility function, in ConfigFile.cpp, but NewHTKMLFReader doesn't need that code...

// Trim - trim white space off the start and end of the string
// str - string to trim
// NOTE: if the entire string is empty, then the string will be set to an empty string
void Trim(std::string& str)
{
    auto found = str.find_first_not_of(" \t");
    if (found == npos)
    {
        str.erase(0);
        return;
    }
    str.erase(0, found);
    found = str.find_last_not_of(" \t");
    if (found != npos)
        str.erase(found + 1);
}
#endif
} } }
