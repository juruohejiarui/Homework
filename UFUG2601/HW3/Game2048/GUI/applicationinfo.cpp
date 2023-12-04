#include "applicationinfo.h"
#include <iostream>

std::string getApplicationDir() {
    QString _res = QCoreApplication::applicationDirPath();
#ifdef __APPLE__
    _res = _res.left(_res.lastIndexOf("Game2048-gui.app"));
#elif __linux__
    _res += "/";
#elif _WIN32
    _res += "/"
#endif
    return _res.toStdString();
}
