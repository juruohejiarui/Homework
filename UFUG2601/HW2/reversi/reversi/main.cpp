#include "mainwindow.h"

#include <QApplication>
#include <QFontDatabase>
#include <iostream>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

//    std::cout << QCoreApplication::applicationDirPath().toStdString() << "/Resources/Fonts/GenshinFont.ttf" << std::endl;
    int fontId = QFontDatabase::addApplicationFont(":/Resources/Fonts/GenshinFont.ttf");

    QString fontName = QFontDatabase::applicationFontFamilies(fontId).at(0);
    QFont font = QFont(fontName, 12);
    QApplication::setFont(font);

    return a.exec();
}
