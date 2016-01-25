//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "htkfeatio.h"
#include "msra_mgram.h"

namespace Microsoft { namespace MSR { namespace CNTK {

MLFDataDeserializer::MLFDataDeserializer(const ConfigParameters& label, size_t elementSize, const HTKDataDeserializer* featureDeserializer, bool frameMode, const std::wstring& name)
    : m_mlfPaths(std::move(ConfigHelper::GetMlfPaths(label))), m_elementSize(elementSize), m_featureDeserializer(featureDeserializer), m_frameMode(frameMode), m_name(name)
{
    ConfigHelper::CheckLabelType(label);

    m_dimension = ConfigHelper::GetLabelDimension(label);
    m_layout = std::make_shared<TensorShape>(m_dimension);

    m_stateListPath = static_cast<std::wstring>(label(L"labelMappingFile", L""));

    // TODO: currently assumes all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files

    // get labels
    const double htktimetoframe = 100000.0; // default is 10ms

    const msra::lm::CSymbolSet* wordTable = nullptr;
    std::map<string, size_t>* symbolTable = nullptr;

    msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence> labels(m_mlfPaths, std::set<wstring>(), m_stateListPath, wordTable, symbolTable, htktimetoframe);

    // Make sure 'msra::asr::htkmlfreader' type has a move constructor
    static_assert(
        std::is_move_constructible<
            msra::asr::htkmlfreader<msra::asr::htkmlfentry,
                                    msra::lattices::lattice::htkmlfwordsequence>>::value,
        "Type 'msra::asr::htkmlfreader' should be move constructible!");

    MLFUtterance description;
    description.m_id = 0;
    description.m_isValid = true; // right now we throw for invalid sequences
    // TODO .chunk, .key

    size_t totalFrames = 0;
    // Have to iterate in the same order as utterances inside the HTK data de-serializer to be aligned.
    for (const auto& u : featureDeserializer->GetUtterances())
    {
        wstring key = u.utterance.key();

        // todo check that actually exists.
        auto l = labels.find(key);
        const auto& labseq = l->second;

        description.sequenceStart = m_classIds.size(); // TODO
        description.m_isValid = true;
        size_t numofframes = 0;
        description.m_id++;

        foreach_index (i, labseq)
        {
            // TODO Why will these yield a run-time error as opposed to making the utterance invalid?
            const auto& e = labseq[i];
            if ((i == 0 && e.firstframe != 0) ||
                (i > 0 && labseq[i - 1].firstframe + labseq[i - 1].numframes != e.firstframe))
            {
                RuntimeError("minibatchutterancesource: labels not in consecutive order MLF in label set: %ls", l->first.c_str());
            }

            if (e.classid >= m_dimension)
            {
                RuntimeError("minibatchutterancesource: class id %d exceeds model output dimension %d in file",
                             static_cast<int>(e.classid),
                             static_cast<int>(m_dimension));
            }

            if (e.classid != static_cast<msra::dbn::CLASSIDTYPE>(e.classid))
            {
                RuntimeError("CLASSIDTYPE has too few bits");
            }

            for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
            {
                m_classIds.push_back(e.classid);
                numofframes++;
            }
        }

        description.m_numberOfSamples = numofframes;
        totalFrames += numofframes;
        m_utterances.push_back(description);
    }

    if (m_frameMode)
    {
        m_frames.reserve(totalFrames);
    }
    else
    {
        m_sequences.reserve(m_utterances.size());
    }

    foreach_index (i, m_utterances)
    {
        if (m_frameMode)
        {
            for (size_t k = 0; k < m_utterances[i].m_numberOfSamples; ++k)
            {
                MLFFrame f;
                f.m_id = m_frames.size();
                f.m_chunkId = 0;
                f.m_numberOfSamples = 1;
                f.index = m_utterances[i].sequenceStart + k;
                assert(m_utterances[i].m_isValid); // TODO
                f.m_isValid = m_utterances[i].m_isValid;
                m_frames.push_back(f);
                m_sequences.push_back(&m_frames[f.m_id]);
            }
        }
        else
        {
            assert(false);
            m_sequences.push_back(&m_utterances[i]);
        }
    }
}

void Microsoft::MSR::CNTK::MLFDataDeserializer::StartEpoch(const EpochConfiguration& /*config*/)
{
    throw std::logic_error("The method or operation is not implemented.");
}

const SequenceDescriptions& Microsoft::MSR::CNTK::MLFDataDeserializer::GetSequenceDescriptions() const
{
    return m_sequences;
}

std::vector<StreamDescriptionPtr> MLFDataDeserializer::GetStreamDescriptions() const
{
    StreamDescriptionPtr stream = std::make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = m_name;
    stream->m_sampleLayout = std::make_shared<TensorShape>(m_dimension);
    stream->m_elementType = m_elementSize == sizeof(float) ? ElementType::tfloat : ElementType::tdouble;
    return std::vector<StreamDescriptionPtr>{stream};
}

std::vector<std::vector<SequenceDataPtr>> MLFDataDeserializer::GetSequencesById(const std::vector<size_t>& ids)
{
    assert(m_frameMode);
    assert(ids.size() == 1);
    auto id = ids[0];

    size_t label = m_classIds[m_frames[id].index];
    DenseSequenceDataPtr r = std::make_shared<DenseSequenceData>();
    if (m_elementSize == sizeof(float))
    {
        float* tmp = new float[m_dimension];
        memset(tmp, 0, m_elementSize * m_dimension);
        tmp[label] = 1;
        r->m_data = tmp;
    }
    else
    {
        double* tmp = new double[m_dimension];
        memset(tmp, 0, m_elementSize * m_dimension);
        tmp[label] = 1;
        r->m_data = tmp;
    }

    r->m_numberOfSamples = m_sequences[id]->m_numberOfSamples;

    std::vector<std::vector<SequenceDataPtr>> result;
    result.push_back(std::vector<SequenceDataPtr>{r});
    return result;
}

void MLFDataDeserializer::RequireChunk(size_t /*chunkIndex*/)
{
}

void MLFDataDeserializer::ReleaseChunk(size_t /*chunkIndex*/)
{
}

const std::vector<MLFUtterance>& MLFDataDeserializer::GetUtterances() const
{
    return m_utterances;
}
} } }
