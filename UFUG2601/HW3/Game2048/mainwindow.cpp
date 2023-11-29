#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    gameBoard = findChild<GameBoard *>("gameBoard");
    actionNewGame = findChild<QAction *>("actionNewGame");
    actionHelp = findChild<QAction *>("actionHelp");

    connect(actionNewGame, SIGNAL(triggered()), SLOT(actionNewGameClicked()));
    connect(actionHelp, SIGNAL(triggered()), SLOT(actionHelpClicked()));
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    gameBoard->resize(((QWidget *)gameBoard->parent())->size());
}
MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::actionNewGameClicked() {
    gameBoard->tryNewGame();
}

void MainWindow::actionHelpClicked() {

}
