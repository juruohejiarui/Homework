#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    gameBoard = findChild<GameBoard *>("gameBoard");
    actionNewGame = findChild<QAction *>("actionNewGame");
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    gameBoard->resize(((QWidget *)gameBoard->parent())->size());
}
MainWindow::~MainWindow()
{
    delete ui;
}

