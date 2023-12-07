#include "applicationinfo.h"
#include <iostream>

std::string getApplicationDir() {
    QString _res = QCoreApplication::applicationDirPath();
#ifdef __APPLE__
    _res += "../../";
#elif __linux__
    _res += "/";
#elif _WIN32
    _res += "/";
#endif
    return _res.toStdString();
}
