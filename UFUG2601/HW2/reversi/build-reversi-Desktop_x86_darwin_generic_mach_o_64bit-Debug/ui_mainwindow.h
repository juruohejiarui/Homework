/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.6.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>
#include <sboard.h>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionNew_Game;
    QAction *actionDark_Mode_2;
    QAction *actionLight_Mode;
    QAction *actionGenshin_Mode;
    QWidget *centralwidget;
    SBoard *widget;
    QMenuBar *menubar;
    QMenu *menuConfig;
    QMenu *menuConfigs;
    QMenu *menuThemes;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(600, 650);
        MainWindow->setMinimumSize(QSize(600, 600));
        actionNew_Game = new QAction(MainWindow);
        actionNew_Game->setObjectName("actionNew_Game");
        actionDark_Mode_2 = new QAction(MainWindow);
        actionDark_Mode_2->setObjectName("actionDark_Mode_2");
        actionLight_Mode = new QAction(MainWindow);
        actionLight_Mode->setObjectName("actionLight_Mode");
        actionGenshin_Mode = new QAction(MainWindow);
        actionGenshin_Mode->setObjectName("actionGenshin_Mode");
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        widget = new SBoard(centralwidget);
        widget->setObjectName("widget");
        widget->setGeometry(QRect(0, 0, 600, 601));
        sizePolicy.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy);
        widget->setMinimumSize(QSize(0, 0));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 600, 22));
        menuConfig = new QMenu(menubar);
        menuConfig->setObjectName("menuConfig");
        menuConfigs = new QMenu(menubar);
        menuConfigs->setObjectName("menuConfigs");
        menuThemes = new QMenu(menuConfigs);
        menuThemes->setObjectName("menuThemes");
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuConfig->menuAction());
        menubar->addAction(menuConfigs->menuAction());
        menuConfig->addAction(actionNew_Game);
        menuConfigs->addAction(menuThemes->menuAction());
        menuThemes->addAction(actionDark_Mode_2);
        menuThemes->addAction(actionLight_Mode);
        menuThemes->addAction(actionGenshin_Mode);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        actionNew_Game->setText(QCoreApplication::translate("MainWindow", "New Game", nullptr));
        actionDark_Mode_2->setText(QCoreApplication::translate("MainWindow", "Dark Mode", nullptr));
        actionLight_Mode->setText(QCoreApplication::translate("MainWindow", "Light Mode", nullptr));
        actionGenshin_Mode->setText(QCoreApplication::translate("MainWindow", "Genshin Mode", nullptr));
        menuConfig->setTitle(QCoreApplication::translate("MainWindow", "Operations", nullptr));
        menuConfigs->setTitle(QCoreApplication::translate("MainWindow", "Configs", nullptr));
        menuThemes->setTitle(QCoreApplication::translate("MainWindow", "Themes", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
