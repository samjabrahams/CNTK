//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // TODO: currently this code moved from the old block randomizer.
    // The class will be further refactored and common based will be extracted
    // with BlockRandomizer.
    // The class represents a transformer that does not randomize input.
    class NoRandomizer : public Transformer
    {
    public:
        NoRandomizer(DataDeserializerPtr deserializer);

        virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;
        virtual void StartEpoch(const EpochConfiguration& config) override;
        virtual Sequences GetNextSequences(size_t count) override;
        virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
        {
            return m_deserializer->GetStreamDescriptions();
        }

    private:
        bool AdvanceToNextPositionForThisWorker();

        // Deserializer and information on the original timeline
        DataDeserializerPtr m_deserializer;
        // Initial timeline.
        SequenceDescriptions m_timeline;
        size_t m_totalNumberOfSamples;

        // Epoch configuration
        EpochConfiguration m_config;
        size_t m_samplePositionInEpoch;
        size_t m_sequencePosition;
    };
}}}
