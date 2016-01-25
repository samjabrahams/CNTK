//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NewHTKMLFReaderShim.cpp: implementation for shim that wraps new HTKMLF reader
//

#include "stdafx.h"
#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#include "htkfeatio.h"      // for reading HTK features
#include "latticearchive.h" // for reading HTK phoneme lattices (MMI training)

#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "Config.h"
#include "NewHTKMLFReaderShim.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

#ifdef __unix__
// TODO probably not needed anymore
#include <limits.h>
#endif

#include "SubstitutingMemoryProvider.h"
#include "CudaMemoryProvider.h"
#include "HeapMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::Init(const ConfigParameters& config)
{
    m_layout = make_shared<MBLayout>();

    assert(config(L"frameMode", true));
    m_memoryProvider = std::make_shared<HeapMemoryProvider>();
    m_packer = std::make_shared<FrameModePacker>(config, m_memoryProvider, sizeof(ElemType));

    intargvector numberOfuttsPerMinibatchForAllEpochs =
        config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{1})));

    auto numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;
    m_layout->Init(numSeqsPerMBForAllEpochs[0], 0);
    m_streams = m_packer->GetStreamDescriptions();
}

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
}

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize*/)
{
    EpochConfiguration config;
    config.m_workerRank = subsetNum;
    config.m_numberOfWorkers = numSubsets;
    config.m_minibatchSizeInSamples = requestedMBSize;
    config.m_totalEpochSizeInSamples = requestedEpochSamples;
    config.m_epochIndex = epoch;

    m_packer->StartEpoch(config);
}

template <class ElemType>
bool NewHTKMLFReaderShim<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    // eldak: Hack.
    int deviceId = matrices.begin()->second->GetDeviceId();
    for (auto mx : matrices)
    {
        if (mx.second->GetDeviceId() != deviceId)
        {
            assert(false);
        }
    }

    Minibatch m = m_packer->ReadMinibatch();
    if (m.m_endOfEpoch)
    {
        return false;
    }

    auto streams = m_packer->GetStreamDescriptions();
    std::map<size_t, wstring> idToName;
    for (auto i : streams)
    {
        idToName.insert(std::make_pair(i->m_id, i->m_name));
    }

    for (int i = 0; i < m.m_data.size(); i++)
    {
        const auto& stream = m.m_data[i];
        const std::wstring& name = idToName[i];
        if (matrices.find(name) == matrices.end())
        {
            continue;
        }

        // Current hack.
        m_layout = stream->m_layout;
        size_t columnNumber = m_layout->GetNumCols();
        size_t rowNumber = m_streams[i]->m_sampleLayout->GetNumElements();

        auto data = reinterpret_cast<const ElemType*>(stream->m_data);
        matrices[name]->SetValue(rowNumber, columnNumber, matrices[name]->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal);
    }

    return !m.m_endOfEpoch;
}

template <class ElemType>
bool NewHTKMLFReaderShim<ElemType>::DataEnd(EndDataType /*endDataType*/)
{
    return false;
}

template <class ElemType>
void NewHTKMLFReaderShim<ElemType>::CopyMBLayoutTo(MBLayoutPtr layout)
{
    layout->CopyFrom(m_layout);
}

template <class ElemType>
size_t NewHTKMLFReaderShim<ElemType>::GetNumParallelSequences()
{
    return m_layout->GetNumParallelSequences(); // (this function is only used for validation anyway)
}

template class NewHTKMLFReaderShim<float>;
template class NewHTKMLFReaderShim<double>;
} } }
