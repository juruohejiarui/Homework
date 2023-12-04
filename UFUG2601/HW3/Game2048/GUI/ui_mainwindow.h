/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>
#include "gameboard.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionResize33;
    QAction *actionResize44;
    QAction *actionResize55;
    QAction *actionResize66;
    QAction *actionResizeCustom;
    QAction *actionThemeClassic;
    QAction *actionThemeBlue;
    QAction *actionNewGame;
    QAction *actionShow_Rank;
    QAction *actionChangePlayer;
    QAction *actionUndo;
    QAction *actionConfiguration_Panel;
    QAction *actionResize88;
    QAction *actionPause;
    QAction *actionBoard;
    QWidget *centralwidget;
    GameBoard *gameBoard;
    QMenuBar *menubar;
    QMenu *menuConfiguration;
    QMenu *menuBoard_Size;
    QMenu *menuTheme;
    QMenu *menuGame;
    QMenu *menuView;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 600);
        MainWindow->setMinimumSize(QSize(500, 500));
        actionResize33 = new QAction(MainWindow);
        actionResize33->setObjectName(QString::fromUtf8("actionResize33"));
        actionResize44 = new QAction(MainWindow);
        actionResize44->setObjectName(QString::fromUtf8("actionResize44"));
        actionResize55 = new QAction(MainWindow);
        actionResize55->setObjectName(QString::fromUtf8("actionResize55"));
        actionResize66 = new QAction(MainWindow);
        actionResize66->setObjectName(QString::fromUtf8("actionResize66"));
        actionResizeCustom = new QAction(MainWindow);
        actionResizeCustom->setObjectName(QString::fromUtf8("actionResizeCustom"));
        actionThemeClassic = new QAction(MainWindow);
        actionThemeClassic->setObjectName(QString::fromUtf8("actionThemeClassic"));
        actionThemeBlue = new QAction(MainWindow);
        actionThemeBlue->setObjectName(QString::fromUtf8("actionThemeBlue"));
        actionNewGame = new QAction(MainWindow);
        actionNewGame->setObjectName(QString::fromUtf8("actionNewGame"));
        actionShow_Rank = new QAction(MainWindow);
        actionShow_Rank->setObjectName(QString::fromUtf8("actionShow_Rank"));
        actionChangePlayer = new QAction(MainWindow);
        actionChangePlayer->setObjectName(QString::fromUtf8("actionChangePlayer"));
        actionUndo = new QAction(MainWindow);
        actionUndo->setObjectName(QString::fromUtf8("actionUndo"));
        actionConfiguration_Panel = new QAction(MainWindow);
        actionConfiguration_Panel->setObjectName(QString::fromUtf8("actionConfiguration_Panel"));
        actionResize88 = new QAction(MainWindow);
        actionResize88->setObjectName(QString::fromUtf8("actionResize88"));
        actionPause = new QAction(MainWindow);
        actionPause->setObjectName(QString::fromUtf8("actionPause"));
        actionBoard = new QAction(MainWindow);
        actionBoard->setObjectName(QString::fromUtf8("actionBoard"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gameBoard = new GameBoard(centralwidget);
        gameBoard->setObjectName(QString::fromUtf8("gameBoard"));
        gameBoard->setGeometry(QRect(0, 0, 800, 556));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(gameBoard->sizePolicy().hasHeightForWidth());
        gameBoard->setSizePolicy(sizePolicy);
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 22));
        menuConfiguration = new QMenu(menubar);
        menuConfiguration->setObjectName(QString::fromUtf8("menuConfiguration"));
        menuBoard_Size = new QMenu(menuConfiguration);
        menuBoard_Size->setObjectName(QString::fromUtf8("menuBoard_Size"));
        menuTheme = new QMenu(menuConfiguration);
        menuTheme->setObjectName(QString::fromUtf8("menuTheme"));
        menuGame = new QMenu(menubar);
        menuGame->setObjectName(QString::fromUtf8("menuGame"));
        menuView = new QMenu(menubar);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuConfiguration->menuAction());
        menubar->addAction(menuGame->menuAction());
        menubar->addAction(menuView->menuAction());
        menuConfiguration->addAction(menuBoard_Size->menuAction());
        menuConfiguration->addAction(menuTheme->menuAction());
        menuConfiguration->addAction(actionChangePlayer);
        menuBoard_Size->addAction(actionResize33);
        menuBoard_Size->addAction(actionResize44);
        menuBoard_Size->addAction(actionResize55);
        menuBoard_Size->addAction(actionResize66);
        menuBoard_Size->addAction(actionResize88);
        menuBoard_Size->addAction(actionResizeCustom);
        menuTheme->addAction(actionThemeClassic);
        menuTheme->addAction(actionThemeBlue);
        menuGame->addAction(actionNewGame);
        menuGame->addAction(actionUndo);
        menuView->addAction(actionShow_Rank);
        menuView->addAction(actionPause);
        menuView->addAction(actionBoard);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        actionResize33->setText(QCoreApplication::translate("MainWindow", "3*3", nullptr));
        actionResize44->setText(QCoreApplication::translate("MainWindow", "4*4", nullptr));
        actionResize55->setText(QCoreApplication::translate("MainWindow", "5*5", nullptr));
        actionResize66->setText(QCoreApplication::translate("MainWindow", "6*6", nullptr));
        actionResizeCustom->setText(QCoreApplication::translate("MainWindow", "Custom Size", nullptr));
        actionThemeClassic->setText(QCoreApplication::translate("MainWindow", "Classic", nullptr));
        actionThemeBlue->setText(QCoreApplication::translate("MainWindow", "Blue", nullptr));
        actionNewGame->setText(QCoreApplication::translate("MainWindow", "New Game", nullptr));
        actionShow_Rank->setText(QCoreApplication::translate("MainWindow", "Rank List", nullptr));
        actionChangePlayer->setText(QCoreApplication::translate("MainWindow", "Player Name", nullptr));
        actionUndo->setText(QCoreApplication::translate("MainWindow", "Undo", nullptr));
        actionConfiguration_Panel->setText(QCoreApplication::translate("MainWindow", "Configuration Panel", nullptr));
        actionResize88->setText(QCoreApplication::translate("MainWindow", "8*8", nullptr));
        actionPause->setText(QCoreApplication::translate("MainWindow", "Pause", nullptr));
        actionBoard->setText(QCoreApplication::translate("MainWindow", "Board", nullptr));
        menuConfiguration->setTitle(QCoreApplication::translate("MainWindow", "Configuration", nullptr));
        menuBoard_Size->setTitle(QCoreApplication::translate("MainWindow", "Board Size", nullptr));
        menuTheme->setTitle(QCoreApplication::translate("MainWindow", "Theme", nullptr));
        menuGame->setTitle(QCoreApplication::translate("MainWindow", "Operation", nullptr));
        menuView->setTitle(QCoreApplication::translate("MainWindow", "View", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
