#include "mainwindow.h"
#include "applicationinfo.h"
#include "dialogs.h"
#include "ui_mainwindow.h"
#include <QInputDialog>
#include <QString>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    gameBoard = findChild<GameBoard *>("gameBoard");

    actionNewGame = findChild<QAction *>("actionNewGame");
    actionUndo = findChild<QAction *>("actionUndo");
    actionChangePlayer = findChild<QAction *>("actionChangePlayer");

    actionResize33 = findChild<QAction *>("actionResize33");
    actionResize44 = findChild<QAction *>("actionResize44");
    actionResize55 = findChild<QAction *>("actionResize55");
    actionResize66 = findChild<QAction *>("actionResize66");
    actionResize88 = findChild<QAction *>("actionResize88");
    actionResizeCustom = findChild<QAction *>("actionResizeCustom");

    actionThemeClassic = findChild<QAction *>("actionThemeClassic");
    actionThemeBlue = findChild<QAction *>("actionThemeBlue");

    connect(actionNewGame, SIGNAL(triggered()), SLOT(actionNewGameClicked()));
    connect(actionUndo, SIGNAL(triggered()), SLOT(actionUndoClicked()));

    connect(actionChangePlayer, SIGNAL(triggered()), SLOT(actionChangePlayerClicked()));

    connect(actionResize33, SIGNAL(triggered()), SLOT(actionResize33Clicked()));
    connect(actionResize44, SIGNAL(triggered()), SLOT(actionResize44Clicked()));
    connect(actionResize55, SIGNAL(triggered()), SLOT(actionResize55Clicked()));
    connect(actionResize66, SIGNAL(triggered()), SLOT(actionResize66Clicked()));
    connect(actionResize88, SIGNAL(triggered()), SLOT(actionResize88Clicked()));
    connect(actionResizeCustom, SIGNAL(triggered()), SLOT(actionResizeCustomClicked()));

    connect(actionThemeClassic, SIGNAL(triggered()), SLOT(actionThemeClassicClicked()));
    connect(actionThemeBlue, SIGNAL(triggered()), SLOT(actionThemeBlueClicked()));
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    gameBoard->resize(((QWidget *)gameBoard->parent())->size());
}
MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::actionChangePlayerClicked() {
    gameBoard->tryChangePlayer();
}

void MainWindow::actionNewGameClicked() {
    gameBoard->tryNewGame();
}

void MainWindow::actionUndoClicked() {
    gameBoard->undo();
}

void MainWindow::closeEvent(QCloseEvent *event) {
    gameBoard->save();
}

void MainWindow::actionResize33Clicked() {
    gameBoard->tryResizeBoard(3, 3);
}

void MainWindow::actionResize44Clicked() {
    gameBoard->tryResizeBoard(4, 4);
}

void MainWindow::actionResize55Clicked() {
    gameBoard->tryResizeBoard(5, 5);
}

void MainWindow::actionResize66Clicked() {
    gameBoard->tryResizeBoard(6, 6);
}

void MainWindow::actionResize88Clicked() {
    gameBoard->tryResizeBoard(8, 8);
}


void MainWindow::actionResizeCustomClicked() {
    gameBoard->tryResizeBoard();
}

void MainWindow::actionThemeClassicClicked() {
    gameBoard->changeTheme("Classic.theme");
}

void MainWindow::actionThemeBlueClicked() {
    gameBoard->changeTheme("Blue.theme");
}
