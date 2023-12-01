#include "applicationinfo.h"
#include <iostream>

std::string getApplicationDir() {
    QString _res = QCoreApplication::applicationDirPath();
#ifdef Q_OS_MACOS
    _res = _res.left(_res.lastIndexOf("Game2048.app"));
#else
    _res += "/";
#endif
    return _res.toStdString();
}
