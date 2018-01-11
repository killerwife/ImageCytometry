#pragma once
#include <string>
#include <vector>

class FileFinder
{
public:
    FileFinder();
    ~FileFinder();

    void GetFileNames(std::string & path, std::vector<std::string>& filenames);
};

