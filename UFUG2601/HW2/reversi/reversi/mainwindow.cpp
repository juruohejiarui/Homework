#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    centralWidget = findChild<QWidget *>("centralwidget");
    sboard = findChild<SBoard *>("sboard");
    newGameItem = findChild<QAction *>("actionNew_Game");
    changeSize8_8Item = findChild<QAction *>("action8_8");
    changeSize8_16Item = findChild<QAction *>("action8_16");
    changeSize16_16Item = findChild<QAction *>("action16_16");

    connect(newGameItem, SIGNAL(triggered()), this, SLOT(newGameItemClicked()));
    connect(changeSize8_8Item, SIGNAL(triggered()), this, SLOT(changeSize8_8ItemClicked()));
    connect(changeSize8_16Item, SIGNAL(triggered()), this, SLOT(changeSize8_16ItemClicked()));
    connect(changeSize16_16Item, SIGNAL(triggered()), this, SLOT(changeSize16_16ItemClicked()));
}

void MainWindow::newGameItemClicked() {
    sboard->tryCreateNewGame();
}

void MainWindow::changeSize8_8ItemClicked() {
    sboard->trySetSize(8, 8);
}

void MainWindow::changeSize8_16ItemClicked() {
    sboard->trySetSize(8, 16);
}

void MainWindow::changeSize16_16ItemClicked() {
    sboard->trySetSize(16, 16);
}
void MainWindow::resizeEvent(QResizeEvent *ev) {
    sboard->resize(centralWidget->size());
}

MainWindow::~MainWindow()
{
    delete ui;
}

