//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

//#include <objbase.h>
#include "Basics.h"    // for attempt()
#include "htkfeatio.h" // for reading HTK features
#include "minibatchsourcehelpers.h"
#ifndef __unix__
#include "ssematrix.h" // TODO: why can it not be removed for Windows as well? At least needs a comment here.
#endif

#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif

namespace msra { namespace dbn {

class FileEvalSource // : public numamodelmanager
{
    const size_t chunksize; // actual block size to perform computation on

    // data FIFO
    std::vector<msra::dbn::matrix> feat;
    std::vector<std::vector<std::vector<float>>> framesMulti; // [t] all feature frames concatenated into a big block
    std::vector<char> boundaryFlags;                          // [t] -1 for first and +1 last frame, 0 else (for augmentneighbors())
    std::vector<size_t> numFrames;                            // [k] number of frames for all appended files
    std::vector<std::vector<unsigned int>> sampPeriods;       // [k] and sample periods (they should really all be the same...)
    std::vector<size_t> vdims;                                // input dimension
    std::vector<size_t> leftcontext;
    std::vector<size_t> rightcontext;
    bool minibatchReady;
    size_t minibatchSize;
    size_t frameIndex;

    void operator=(const FileEvalSource &);

private:
    void Clear() // empty the FIFO
    {
        foreach_index (i, vdims)
        {
            framesMulti[i].clear();
            sampPeriods[i].clear();
        }
        boundaryFlags.clear();
        numFrames.clear();
        minibatchReady = false;
        frameIndex = 0;
    }

public:
    FileEvalSource(std::vector<size_t> vdims, std::vector<size_t> leftcontext, std::vector<size_t> rightcontext, size_t chunksize)
        : vdims(vdims), leftcontext(leftcontext), rightcontext(rightcontext), chunksize(chunksize)
    {
        foreach_index (i, vdims)
        {
            msra::dbn::matrix thisfeat;
            std::vector<std::vector<float>> frames; // [t] all feature frames concatenated into a big block

            frames.reserve(chunksize * 2);
            framesMulti.push_back(frames);
            //framesmulti[i].reserve (chunksize * 2);

            thisfeat.resize(vdims[i], chunksize);
            feat.push_back(thisfeat);

            sampPeriods.push_back(std::vector<unsigned int>());
            //feat[i].resize(vdims[i],chunksize); // initialize to size chunksize
        }
    }

    // append data to chunk
    template <class MATRIX>
    void AddFile(const MATRIX &feat, const string & /*featkind*/, unsigned int sampPeriod, size_t index)
    {
        // append to frames; also expand neighbor frames
        if (feat.cols() < 2)
            RuntimeError("evaltofile: utterances < 2 frames not supported");
        foreach_column (t, feat)
        {
            std::vector<float> v(&feat(0, t), &feat(0, t) + feat.rows());
            framesMulti[index].push_back(v);
            if (index == 0)
                boundaryFlags.push_back((t == 0) ? -1 : (t == feat.cols() - 1) ? +1 : 0);
        }
        if (index == 0)
            numFrames.push_back(feat.cols());

        sampPeriods[index].push_back(sampPeriod);
    }

    void CreateEvalMinibatch()
    {
        foreach_index (i, framesMulti)
        {
            const size_t framesInBlock = framesMulti[i].size();
            feat[i].resize(vdims[i], framesInBlock); // input features for whole utt (col vectors)
            // augment the features
            size_t leftextent, rightextent;
            // page in the needed range of frames
            if (leftcontext[i] == 0 && rightcontext[i] == 0)
            {
                leftextent = rightextent = augmentationextent(framesMulti[i][0].size(), vdims[i]);
            }
            else
            {
                leftextent = leftcontext[i];
                rightextent = rightcontext[i];
            }

            //msra::dbn::augmentneighbors(framesMulti[i], boundaryFlags, 0, leftcontext[i], rightcontext[i],)
            msra::dbn::augmentneighbors(framesMulti[i], boundaryFlags, leftextent, rightextent, 0, framesInBlock, feat[i]);
        }
        minibatchReady = true;
    }

    void SetMinibatchSize(size_t mbSize)
    {
        minibatchSize = mbSize;
    }
    msra::dbn::matrix ChunkOfFrames(size_t index)
    {
        assert(minibatchReady);
        assert(index <= feat.size());
        return feat[index];
    }

    bool IsMinibatchReady()
    {
        return minibatchReady;
    }

    size_t CurrentFileSize()
    {
        return framesMulti[0].size();
    }
    void FlushInput()
    {
        CreateEvalMinibatch();
    }
    void Reset()
    {
        Clear();
    }
};
};
};
