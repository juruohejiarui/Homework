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
}

void MainWindow::resizeEvent(QResizeEvent *ev) {
    sboard->resize(centralWidget->size());
}

MainWindow::~MainWindow()
{
    delete ui;
}

