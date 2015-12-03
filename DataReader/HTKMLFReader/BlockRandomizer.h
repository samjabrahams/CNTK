//
// <copyright file="BlockRandomizer.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// BlockRandomizer.h -- interface of the block randomizer
//

#pragma once

#include "Basics.h"                  // for attempt()
#include "htkfeatio.h"                  // for htkmlfreader
#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "minibatchsourcehelpers.h"
#include "minibatchiterator.h"
#include "biggrowablevectors.h"
#include "ssematrix.h"
#include "unordered_set"

namespace msra { namespace dbn {

    // data store (incl. paging in/out of features and lattices)
    struct utterancedesc            // data descriptor for one utterance
    {
        msra::asr::htkfeatreader::parsedpath parsedpath;    // archive filename and frame range in that file
        size_t classidsbegin;       // index into allclassids[] array (first frame)

        utterancedesc (msra::asr::htkfeatreader::parsedpath&& ppath, size_t classidsbegin) : parsedpath (std::move(ppath)), classidsbegin (classidsbegin) {}

        const wstring & logicalpath() const { return parsedpath; /*type cast will return logical path*/ }
        size_t numframes() const { return parsedpath.numframes(); }
        wstring key() const                           // key used for looking up lattice (not stored to save space)
        {
#ifdef _MSC_VER
            static const wstring emptywstring;
            static const wregex deleteextensionre (L"\\.[^\\.\\\\/:]*$");
            return regex_replace (logicalpath(), deleteextensionre, emptywstring);  // delete extension (or not if none)
#else
            return removeExtension(logicalpath());
#endif
        }
    };

    // Make sure type 'utterancedesc' has a move constructor
    static_assert(std::is_move_constructible<utterancedesc>::value, "Type 'utterancedesc' should be move constructible!");

    struct utterancechunkdata       // data for a chunk of utterances
    {
        std::vector<utterancedesc> utteranceset;    // utterances in this set
        size_t numutterances() const { return utteranceset.size(); }

        std::vector<size_t> firstframes;    // [utteranceindex] first frame for given utterance
        mutable msra::dbn::matrix frames;   // stores all frames consecutively (mutable since this is a cache)
        size_t totalframes;         // total #frames for all utterances in this chunk
        mutable std::vector<shared_ptr<const latticesource::latticepair>> lattices;   // (may be empty if none)

        // construction
        utterancechunkdata() : totalframes (0) {}
        void push_back (utterancedesc &&/*destructive*/ utt)
        {
            if (isinram())
                LogicError("utterancechunkdata: frames already paged into RAM--too late to add data");
            firstframes.push_back (totalframes);
            totalframes += utt.numframes();
            utteranceset.push_back (std::move(utt));
        }

        // accessors to an utterance's data
        size_t numframes (size_t i) const { return utteranceset[i].numframes(); }
        size_t getclassidsbegin (size_t i) const { return utteranceset[i].classidsbegin; }
        msra::dbn::matrixstripe getutteranceframes (size_t i) const // return the frame set for a given utterance
        {
            if (!isinram())
                LogicError("getutteranceframes: called when data have not been paged in");
            const size_t ts = firstframes[i];
            const size_t n = numframes(i);
            return msra::dbn::matrixstripe (frames, ts, n);
        }
        shared_ptr<const latticesource::latticepair> getutterancelattice (size_t i) const // return the frame set for a given utterance
        {
            if (!isinram())
                LogicError("getutterancelattice: called when data have not been paged in");
            return lattices[i];
        }

        // paging
        // test if data is in memory at the moment
        bool isinram() const { return !frames.empty(); }
        // page in data for this chunk
        // We pass in the feature info variables by ref which will be filled lazily upon first read
        void requiredata (string & featkind, size_t & featdim, unsigned int & sampperiod, const latticesource & latticesource, int verbosity=0) const
        {
            if (numutterances() == 0)
                LogicError("requiredata: cannot page in virgin block");
            if (isinram())
                LogicError("requiredata: called when data is already in memory");
            try             // this function supports retrying since we read from the unrealible network, i.e. do not return in a broken state
            {
                msra::asr::htkfeatreader reader;    // feature reader (we reinstantiate it for each block, i.e. we reopen the file actually)
                // if this is the first feature read ever, we explicitly open the first file to get the information such as feature dimension
                if (featdim == 0)
                {
                    reader.getinfo (utteranceset[0].parsedpath, featkind, featdim, sampperiod);
                    fprintf(stderr, "requiredata: determined feature kind as %llu-dimensional '%s' with frame shift %.1f ms\n",
                        featdim, featkind.c_str(), sampperiod / 1e4);
                }
                // read all utterances; if they are in the same archive, htkfeatreader will be efficient in not closing the file
                frames.resize (featdim, totalframes);
                if (!latticesource.empty())
                    lattices.resize (utteranceset.size());
                foreach_index (i, utteranceset)
                {
                    //fprintf (stderr, ".");
                    // read features for this file
                    auto uttframes = getutteranceframes (i);    // matrix stripe for this utterance (currently unfilled)
                    reader.read (utteranceset[i].parsedpath, (const string &) featkind, sampperiod, uttframes);  // note: file info here used for checkuing only
                    // page in lattice data
                    if (!latticesource.empty())
                        latticesource.getlattices (utteranceset[i].key(), lattices[i], uttframes.cols());
                }
                //fprintf (stderr, "\n");
                if (verbosity)
                    fprintf (stderr, "requiredata: %d utterances read\n", (int)utteranceset.size());
            }
            catch (...)
            {
                releasedata();
                throw;
            }
        }
        // page out data for this chunk
        void releasedata() const
        {
            if (numutterances() == 0)
                LogicError("releasedata: cannot page out virgin block");
            if (!isinram())
                LogicError("releasedata: called when data is not memory");
            // release frames
            frames.resize (0, 0);
            // release lattice data
            lattices.clear();
        }
    };

    class BlockRandomizer
    {
        int verbosity;
        bool framemode;
        size_t _totalframes;
        size_t numutterances;
        size_t randomizationrange; // parameter remembered; this is the full window (e.g. 48 hours), not the half window

        size_t currentsweep;            // randomization is currently cached for this sweep; if it changes, rebuild all below
        struct chunk                    // chunk as used in actual processing order (randomized sequence)
        {
            // the underlying chunk (as a non-indexed reference into the chunk set)
            std::vector<utterancechunkdata>::const_iterator uttchunkdata;
            const utterancechunkdata & getchunkdata() const { return *uttchunkdata; }
            size_t numutterances() const { return uttchunkdata->numutterances(); }
            size_t numframes() const { return uttchunkdata->totalframes; }

            // position in utterance-position space
            size_t utteranceposbegin;
            size_t utteranceposend() const { return utteranceposbegin + numutterances(); }

            // position on global time line
            size_t globalts;            // start frame on global timeline (after randomization)
            size_t globalte() const { return globalts + numframes(); }

            // randomization range limits
            // TODO only need to maintain for first feature stream
            size_t windowbegin;         // randomizedchunk index of earliest chunk that utterances in here can be randomized with
            size_t windowend;           // and end index [windowbegin, windowend)
            chunk(std::vector<utterancechunkdata>::const_iterator uttchunkdata,
                size_t utteranceposbegin,
                size_t globalts)
                : uttchunkdata(uttchunkdata)
                , utteranceposbegin(utteranceposbegin)
                , globalts(globalts) {}
        };
        // TODO no need to keep this for every feature stream - order is the same for all
        std::vector<std::vector<chunk>> randomizedchunks;  // utterance chunks after being brought into random order (we randomize within a rolling window over them)

    public:
        struct sequenceref              // described a sequence to be randomized (in frame mode, a single frame; a full utterance otherwise)
        {
            size_t chunkindex;          // lives in this chunk (index into randomizedchunks[])
            size_t utteranceindex;      // utterance index in that chunk
            size_t numframes;           // (cached since we cannot directly access the underlying data from here)
            size_t globalts;            // start frame in global space after randomization (for mapping frame index to utterance position)
            size_t frameindex;          // 0 for utterances

            // TODO globalts - sweep cheaper?
            size_t globalte() const { return globalts + numframes; }            // end frame

            sequenceref()
                : chunkindex (0)
                , utteranceindex (0)
                , frameindex (0)
                , globalts (SIZE_MAX)
                , numframes (0) {}
            sequenceref (size_t chunkindex, size_t utteranceindex, size_t frameindex = 0)
                : chunkindex (chunkindex)
                , utteranceindex (utteranceindex)
                , frameindex (frameindex)
                , globalts (SIZE_MAX)
                , numframes (0) {}

            // TODO globalts and numframes only set after swapping, wouldn't need to swap them
            // TODO old frameref was more tighly packed (less fields, smaller frameindex and utteranceindex). We need to bring these space optimizations back.
        };

    private:
        // TODO rename
        std::vector<sequenceref> randomizedsequencerefs;          // [pos] randomized utterance ids
        std::unordered_map<size_t, size_t> randomizedutteranceposmap;     // [globalts] -> pos lookup table

        struct positionchunkwindow       // chunk window required in memory when at a certain position, for controlling paging
        {
            std::vector<chunk>::iterator definingchunk;       // the chunk in randomizedchunks[] that defined the utterance position of this utterance
            size_t windowbegin() const { return definingchunk->windowbegin; }
            size_t windowend() const { return definingchunk->windowend; }
            bool isvalidforthisposition (const sequenceref & sequence) const
            {
                return sequence.chunkindex >= windowbegin() && sequence.chunkindex < windowend(); // check if 'sequence' lives in is in allowed range for this position
                // TODO by construction sequences cannot span chunks (check again)
            }

            positionchunkwindow (std::vector<chunk>::iterator definingchunk) : definingchunk (definingchunk) {}
        };
        std::vector<positionchunkwindow> positionchunkwindows;      // [utterance position] -> [windowbegin, windowend) for controlling paging

        template<typename VECTOR> static void randomshuffle (VECTOR & v, size_t randomseed);

    public:
        BlockRandomizer(int verbosity, bool framemode, size_t totalframes, size_t numutterances, size_t randomizationrange)
            : verbosity(verbosity)
            , framemode(framemode)
            , _totalframes(totalframes)
            , numutterances(numutterances)
            , randomizationrange(randomizationrange)
            , currentsweep(SIZE_MAX)
        {
        }

        // big long helper to update all cached randomization information
        // This is a rather complex process since we randomize on two levels:
        //  - chunks of consecutive data in the feature archive
        //  - within a range of chunks that is paged into RAM
        //     - utterances (in utt mode), or
        //     - frames (in frame mode)
        // The 'globalts' parameter is the start time that triggered the rerandomization; it is NOT the base time of the randomized area.
        size_t lazyrandomization(
            const size_t globalts,
            const std::vector<std::vector<utterancechunkdata>> & allchunks);

        size_t chunkForFramePos(const size_t t) const  // find chunk for a given frame position
        {
            //inspect chunk of first feature stream only
            auto iter = std::lower_bound(
                randomizedchunks[0].begin(),
                randomizedchunks[0].end(),
                t,
                [&](const chunk & chunk, size_t t) { return chunk.globalte() <= t; });
            const size_t chunkindex = iter - randomizedchunks[0].begin();
            if (t < randomizedchunks[0][chunkindex].globalts || t >= randomizedchunks[0][chunkindex].globalte())
                LogicError("chunkForFramePos: dude, learn STL!");
            return chunkindex;
        }

        // TODO instead give back original chunk index, drop the streamIndex
        const utterancechunkdata & getChunkData(size_t streamIndex, size_t randomizedChunkIndex)
        {
            assert(streamIndex < randomizedchunks.size());
            assert(randomizedChunkIndex < randomizedchunks[streamIndex].size());
            return randomizedchunks[streamIndex][randomizedChunkIndex].getchunkdata();
        }

        size_t getChunkWindowBegin(size_t randomizedChunkIndex)
        {
            const size_t streamIndex = 0;
            assert(randomizedChunkIndex < randomizedchunks[streamIndex].size());
            return randomizedchunks[streamIndex][randomizedChunkIndex].windowbegin;
        }

        size_t getChunkWindowEnd(size_t randomizedChunkIndex)
        {
            const size_t streamIndex = 0;
            assert(randomizedChunkIndex < randomizedchunks[streamIndex].size());
            return randomizedchunks[streamIndex][randomizedChunkIndex].windowend;
        }

        size_t getNumSequences()
        {
            return randomizedsequencerefs.size();
        }

        const sequenceref & getSequenceRef(size_t sequenceIndex)
        {
            assert(sequenceIndex < randomizedsequencerefs.size());
            return randomizedsequencerefs[sequenceIndex];
        }
    };

} }
