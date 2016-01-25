//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataWriter.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"

#include "htkfeatio.h" // for reading HTK features

#define DATAWRITER_EXPORTS
#include "DataWriter.h"
#include "NewHTKMLFWriter.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void DATAWRITER_API GetWriter(IDataWriter<ElemType>** pwriter)
{
    *pwriter = new NewHTKMLFWriter<ElemType>();
}

extern "C" DATAWRITER_API void GetWriterF(IDataWriter<float>** pwriter)
{
    GetWriter(pwriter);
}
extern "C" DATAWRITER_API void GetWriterD(IDataWriter<double>** pwriter)
{
    GetWriter(pwriter);
}

template <class ElemType>
template <class ConfigRecordType>
void DataWriter<ElemType>::InitFromConfig(const ConfigRecordType& writerConfig)
{
    m_dataWriter = new NewHTKMLFWriter<ElemType>();
    m_dataWriter->Init(writerConfig);
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void DataWriter<ElemType>::Destroy()
{
    delete m_dataWriter;
    m_dataWriter = NULL;
}

// DataWriter Constructor
// config - [in] configuration data for the data writer
template <class ElemType>
template <class ConfigRecordType>
DataWriter<ElemType>::DataWriter(const ConfigRecordType& config)
{
    Init(config);
}

// destructor - cleanup temp files, etc.
template <class ElemType>
DataWriter<ElemType>::~DataWriter()
{
    delete m_dataWriter;
    m_dataWriter = NULL;
}

// GetSections - Get the sections of the file
// sections - a map of section name to section. Data specifications from config file will be used to determine where and how to save data
template <class ElemType>
void DataWriter<ElemType>::GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections)
{
    m_dataWriter->GetSections(sections);
}

// SaveData - save data in the file/files
// recordStart - Starting record number
// matrices - a map of section name (section:subsection) to data pointer. Data specifications from config file will be used to determine where and how to save data
// numRecords - number of records we are saving, can be zero if not applicable
// datasetSize - Size of the dataset
// byteVariableSized - for variable sized data, size of current block to be written, zero when not used, or ignored if not variable sized data
template <class ElemType>
bool DataWriter<ElemType>::SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized)
{
    return m_dataWriter->SaveData(recordStart, matrices, numRecords, datasetSize, byteVariableSized);
}

// SaveMapping - save a map into the file
// saveId - name of the section to save into (section:subsection format)
// labelMapping - map we are saving to the file
template <class ElemType>
void DataWriter<ElemType>::SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping)
{
    m_dataWriter->SaveMapping(saveId, labelMapping);
}

//The explicit instantiation
template class DataWriter<double>;
template class DataWriter<float>;
} } }
