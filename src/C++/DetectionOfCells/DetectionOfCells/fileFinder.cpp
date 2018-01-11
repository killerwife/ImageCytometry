#include "fileFinder.h"
#include <windows.h>
#include <fstream>

FileFinder::FileFinder()
{
}


FileFinder::~FileFinder()
{
}

void FileFinder::GetFileNames(std::string& path, std::vector<std::string>& filenames)
{
    std::string temp = path + "\\*.*";
    std::wstring stemp = std::wstring(temp.begin(), temp.end());
    LPCWSTR sw = stemp.c_str();
    HANDLE hFind;
    WIN32_FIND_DATA data;
    hFind = FindFirstFile(sw, &data);
    int i = 0;
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (i >= 2) // skip . and ..
            {
                stemp = data.cFileName;
                filenames.push_back(std::string(stemp.begin(), stemp.end()));
            }
            i++;
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
}
