#include "mainwindow.h"

#include <QApplication>
#include <cstdlib>
#include <QFontDatabase>
#include <iostream>

int main(int argc, char *argv[])
{
    setbuf(stdout, 0);
    srand(time(NULL));
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    int fontId = QFontDatabase::addApplicationFont(":/Resources/Fonts/GenshinFont.ttf");
    QString fontName = QFontDatabase::applicationFontFamilies(fontId).at(0);
    QFont font = QFont(fontName, 12);
    QApplication::setFont(font);


    return a.exec();
}
