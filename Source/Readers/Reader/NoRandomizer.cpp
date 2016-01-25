//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "NoRandomizer.h"
#include "DataReader.h"

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P) (P)
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

NoRandomizer::NoRandomizer(DataDeserializerPtr deserializer)
    : m_deserializer(deserializer),
      m_sequencePosition(0),
      m_samplePositionInEpoch(SIZE_MAX), 
      m_totalNumberOfSamples(0)
{
    assert(deserializer != nullptr);

    m_timeline = m_deserializer->GetSequenceDescriptions();
    for (const auto& seqDesc : m_timeline)
    {
        assert(seqDesc->m_numberOfSamples == 1);
        m_totalNumberOfSamples += seqDesc->m_numberOfSamples;
    }
}

void NoRandomizer::Initialize(TransformerPtr, const ConfigParameters&)
{
}

void NoRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_deserializer->StartEpoch(config);
    m_config = config;

    // TODO: check partial minibatches.
    if (m_config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_config.m_totalEpochSizeInSamples = m_totalNumberOfSamples;
    }

    m_samplePositionInEpoch = 0;
    size_t timeframe = m_config.m_totalEpochSizeInSamples * config.m_epochIndex;
    assert(timeframe != SIZE_MAX); // used as special value for init

    // TODO: This works only for sample mode.
    m_sequencePosition = timeframe % m_totalNumberOfSamples;
};

Sequences NoRandomizer::GetNextSequences(size_t count)
{
    assert(m_samplePositionInEpoch != SIZE_MAX);

    bool endOfEpoch = false;
    std::vector<size_t> originalIds;
    while (originalIds.size() < count)
    {
        endOfEpoch = AdvanceToNextPositionForThisWorker();
        if (endOfEpoch)
        {
            break;
        }

        assert(m_sequencePosition < m_timeline.size());
        const auto& sequence = m_timeline[m_sequencePosition];
        originalIds.push_back(sequence->m_id);
        m_samplePositionInEpoch += sequence->m_numberOfSamples;
        m_sequencePosition++;
    };

    Sequences result;
    result.m_endOfEpoch = endOfEpoch;

    if (originalIds.size() == 0)
    {
        return result;
    }

    result.m_data = m_deserializer->GetSequencesById(originalIds);
    return result;
}

bool NoRandomizer::AdvanceToNextPositionForThisWorker()
{
    while (m_samplePositionInEpoch < m_config.m_totalEpochSizeInSamples)
    {
        m_sequencePosition = m_sequencePosition % m_timeline.size();

        const auto& sequence = m_timeline[m_sequencePosition];
        if ((sequence->m_chunkId % m_config.m_numberOfWorkers) == m_config.m_workerRank)
        {
            // Got one
            break;
        }

        m_samplePositionInEpoch += sequence->m_numberOfSamples;
        m_sequencePosition++;
    }

    return m_config.m_totalEpochSizeInSamples <= m_samplePositionInEpoch;
}

}}}
