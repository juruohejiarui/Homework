QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    applicationinfo.cpp \
    ../Core/configuration.cpp \
    dialogs.cpp \
    gameboard.cpp \
    ../Core/gamestate.cpp \
    ../Core/gamestatepackage.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    applicationinfo.h \
    ../Core/configuration.h \
    dialogs.h \
    gameboard.h \
    ../Core/gamestate.h \
    ../Core/gamestatepackage.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    Resources/Fonts/fonts.qrc
