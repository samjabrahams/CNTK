//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "Config.h" // for ConfigParameters

namespace Microsoft { namespace MSR { namespace CNTK {

class Bundler : public DataDeserializer
{
    void operator=(const Bundler& other); // non-assignable

    bool m_framemode; // true -> actually return frame-level randomized frames (not possible in lattice mode)
    int m_verbosity;

    size_t m_totalframes; // total frames (same as classids.size() if we have labels)
    size_t m_chunksinram; // (for diagnostics messages)

    std::vector<StreamDescriptionPtr> m_streams;
    std::vector<DataDeserializerPtr> m_deserializers;
    DataDeserializerPtr m_driver;

public:
    Bundler(const ConfigParameters& readerConfig, bool framemode, int verbosity,
            DataDeserializerPtr driver, std::vector<DataDeserializerPtr> deserializers);

    virtual void StartEpoch(const EpochConfiguration& config) override;

    virtual const SequenceDescriptions& GetSequenceDescriptions() const override;
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;
    virtual std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) override;
    virtual void RequireChunk(size_t chunkindex) override;
    virtual void ReleaseChunk(size_t chunkIndex) override;
};

std::vector<DataDeserializerPtr> CreateDeserializers(const ConfigParameters& readerConfig,
                                                     bool framemode,
                                                     size_t elementSize);
} } }
