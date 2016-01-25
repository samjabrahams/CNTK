//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "HTKDataDeserializer.h"
#include "ConfigHelper.h"
#include "Basics.h" // for attempt()
#include "minibatchsourcehelpers.h"
#include <numeric>

namespace Microsoft { namespace MSR { namespace CNTK {

HTKDataDeserializer::HTKDataDeserializer(const ConfigParameters& feature, size_t elementSize, bool frameMode, const std::wstring& featureName)
    : m_featureFiles(std::move(ConfigHelper::GetFeaturePaths(feature))), m_elementSize(elementSize), m_featdim(0), m_sampperiod(0), m_verbosity(0), m_frameMode(frameMode), m_featureName(featureName)
{
    ConfigHelper::CheckFeatureType(feature);

    auto context = ConfigHelper::GetContextWindow(feature);

    m_dimension = ConfigHelper::GetFeatureDimension(feature);
    m_dimension = m_dimension * (1 + context.first + context.second);

    m_layout = std::make_shared<TensorShape>(m_dimension);

    size_t numSequences = m_featureFiles.size();

    m_utterances.reserve(numSequences);

    m_context = ConfigHelper::GetContextWindow(feature);

    size_t totalFrames = 0;
    foreach_index (i, m_featureFiles)
    {
        utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(m_featureFiles[i]), 0);
        const size_t uttframes = utterance.numframes(); // will throw if frame bounds not given --required to be given in this mode

        Utterance description(std::move(utterance));
        description.m_id = i;
        // description.chunkId, description.key // TODO

        // we need at least 2 frames for boundary markers to work
        if (uttframes < 2)
        {
            fprintf(stderr, "minibatchutterancesource: skipping %d-th file (%d frames) because it has less than 2 frames: %ls\n",
                    i, static_cast<int>(uttframes), utterance.key().c_str());
            description.m_numberOfSamples = 0;
            description.m_isValid = false;
        }
        else
        {
            description.m_numberOfSamples = uttframes;
            description.m_isValid = true;
        }

        m_utterances.push_back(description);
        totalFrames += description.m_numberOfSamples;
    }

    size_t totalSize = std::accumulate(
        m_utterances.begin(),
        m_utterances.end(),
        static_cast<size_t>(0),
        [](size_t sum, const Utterance& s)
        {
            return s.m_numberOfSamples + sum;
        });

    // distribute them over chunks
    // We simply count off frames until we reach the chunk size.
    // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.
    const size_t framespersec = 100;                   // we just assume this; our efficiency calculation is based on this
    const size_t chunkframes = 15 * 60 * framespersec; // number of frames to target for each chunk

    // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
    // When paging chunk by chunk, chunk size ~14 MB.

    m_chunks.resize(0);
    m_chunks.reserve(totalSize / chunkframes);

    int chunkId = -1;
    foreach_index (i, m_utterances)
    {
        // if exceeding current entry--create a new one
        // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
        if (m_chunks.empty() || m_chunks.back().totalframes > chunkframes || m_chunks.back().numutterances() >= 65535)
        {
            // TODO > instead of >= ? if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= frameref::maxutterancesperchunk)
            m_chunks.push_back(chunkdata());
            chunkId++;
        }

        // append utterance to last chunk
        chunkdata& currentchunk = m_chunks.back();
        m_utterances[i].indexInsideChunk = currentchunk.numutterances();
        currentchunk.push_back(&m_utterances[i].utterance); // move it out from our temp array into the chunk
        m_utterances[i].m_chunkId = chunkId;
        // TODO: above push_back does not actually 'move' because the internal push_back does not accept that
    }

    fprintf(stderr, "minibatchutterancesource: %d utterances grouped into %d chunks, av. chunk size: %.1f utterances, %.1f frames\n",
            static_cast<int>(m_utterances.size()),
            static_cast<int>(m_chunks.size()),
            m_utterances.size() / (double) m_chunks.size(),
            totalSize / (double) m_chunks.size());
    // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.

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
                Frame f(&m_utterances[i]);
                f.m_id = m_frames.size();
                f.m_chunkId = m_utterances[i].m_chunkId;
                f.m_numberOfSamples = 1;
                f.frameIndexInUtterance = k;
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

void HTKDataDeserializer::StartEpoch(const EpochConfiguration& /*config*/)
{
    throw std::logic_error("The method or operation is not implemented.");
}

const SequenceDescriptions& HTKDataDeserializer::GetSequenceDescriptions() const
{
    return m_sequences;
}

std::vector<StreamDescriptionPtr> HTKDataDeserializer::GetStreamDescriptions() const
{
    StreamDescriptionPtr stream = std::make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = m_featureName;
    stream->m_sampleLayout = std::make_shared<TensorShape>(m_dimension);
    stream->m_elementType = m_elementSize == sizeof(float) ? ElementType::tfloat : ElementType::tdouble;
    return std::vector<StreamDescriptionPtr>{stream};
}

class matrixasvectorofvectors // wrapper around a matrix that views it as a vector of column vectors
{
    void operator=(const matrixasvectorofvectors&); // non-assignable
    msra::dbn::matrixbase& m;

public:
    matrixasvectorofvectors(msra::dbn::matrixbase& m)
        : m(m)
    {
    }
    size_t size() const
    {
        return m.cols();
    }
    const_array_ref<float> operator[](size_t j) const
    {
        return array_ref<float>(&m(0, j), m.rows());
    }
};

std::vector<std::vector<SequenceDataPtr>> HTKDataDeserializer::GetSequencesById(const std::vector<size_t>& ids)
{
    assert(ids.size() == 1); // TODO
    auto id = ids[0];

    if (m_frameMode)
    {
        const auto& frame = m_frames[id];
        const auto& utterance = *frame.u;

        msra::dbn::matrix feat;
        feat.resize(m_dimension, 1);

        const std::vector<char> noboundaryflags; // dummy

        const auto& chunkdata = m_chunks[utterance.m_chunkId];

        auto uttframes = chunkdata.getutteranceframes(utterance.indexInsideChunk);
        matrixasvectorofvectors uttframevectors(uttframes); // (wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors())

        size_t leftextent, rightextent;
        // page in the needed range of frames
        if (m_context.first == 0 && m_context.second == 0)
        {
            // should this be moved up?
            leftextent = rightextent = msra::dbn::augmentationextent(uttframevectors[0].size(), m_dimension);
        }
        else
        {
            leftextent = m_context.first;
            rightextent = m_context.second;
        }

        msra::dbn::augmentneighbors(uttframevectors, noboundaryflags, frame.frameIndexInUtterance, leftextent, rightextent, feat, 0);

        DenseSequenceDataPtr r = std::make_shared<DenseSequenceData>();
        r->m_numberOfSamples = utterance.m_numberOfSamples;

        const msra::dbn::matrixstripe featOri = msra::dbn::matrixstripe(feat, 0, feat.cols());
        const size_t dimensions = featOri.rows();
        const void* tmp = &featOri(0, 0);

        // eldak: this should not be allocated each time.
        void* buffer = nullptr;
        if (m_elementSize == sizeof(float))
        {
            buffer = new float[featOri.rows()];
        }
        else
        {
            buffer = new double[featOri.rows()];
        }

        memset(buffer, 0, m_elementSize * dimensions);
        memcpy_s(buffer, m_elementSize * dimensions, tmp, m_elementSize * dimensions);
        r->m_data = buffer;

        std::vector<std::vector<SequenceDataPtr>> result;
        result.push_back(std::vector<SequenceDataPtr>{r});
        return result;
    }
    else
    {
        assert(false);
        return std::vector<std::vector<SequenceDataPtr>>();
    }
}

void HTKDataDeserializer::RequireChunk(size_t chunkIndex)
{
    auto& chunkdata = m_chunks[chunkIndex];
    if (chunkdata.isinram())
    {
        return;
    }

    msra::util::attempt(5, [&]() // (reading from network)
                        {
                            std::unordered_map<std::string, size_t> empty;
                            msra::dbn::latticesource lattices(
                                std::pair<std::vector<std::wstring>, std::vector<std::wstring>>(),
                                empty,
                                std::wstring());
                            chunkdata.requiredata(m_featKind, m_featdim, m_sampperiod, lattices, m_verbosity);
                        });

    m_chunksinram++;
}

void HTKDataDeserializer::ReleaseChunk(size_t chunkIndex)
{
    auto& chunkdata = m_chunks[chunkIndex];
    if (chunkdata.isinram())
    {
        chunkdata.releasedata();
        m_chunksinram--;
    }
}

const std::vector<Utterance>& HTKDataDeserializer::GetUtterances() const
{
    return m_utterances;
}
} } }
