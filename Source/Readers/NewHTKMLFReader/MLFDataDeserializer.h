//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "HTKDataDeserializer.h"
#include "biggrowablevectors.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct MLFUtterance : public SequenceDescription
{
    // Where the sequence is stored in m_classIds
    size_t sequenceStart;
};

struct MLFFrame : public SequenceDescription
{
    // Where the sequence is stored in m_classIds
    size_t index;
};

class MLFDataDeserializer : public DataDeserializer
{
    size_t m_dimension;
    TensorShapePtr m_layout;
    std::wstring m_stateListPath;
    std::vector<std::wstring> m_mlfPaths;
    const HTKDataDeserializer* m_featureDeserializer;

    // [classidsbegin+t] concatenation of all state sequences
    msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> m_classIds;

    std::vector<MLFUtterance> m_utterances;
    std::vector<MLFFrame> m_frames;

    SequenceDescriptions m_sequences;
    size_t m_elementSize;
    bool m_frameMode;
    std::wstring m_name;

public:
    MLFDataDeserializer(const ConfigParameters& label, size_t elementSize, const HTKDataDeserializer* featureDeserializer, bool framMode, const std::wstring& featureName);

    virtual void StartEpoch(const EpochConfiguration& config) override;

    virtual const SequenceDescriptions& GetSequenceDescriptions() const override;

    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    virtual std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) override;

    virtual void RequireChunk(size_t chunkIndex) override;

    virtual void ReleaseChunk(size_t chunkIndex) override;

public:
    const std::vector<MLFUtterance>& GetUtterances() const;
};

typedef std::shared_ptr<MLFDataDeserializer> MLFDataDeserializerPtr;
} } }
