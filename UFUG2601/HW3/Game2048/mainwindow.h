#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QKeyEvent>
#include "gameboard.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
    GameBoard *gameBoard;
    QAction *actionNewGame, *action33, *action44, *action55, *action66, *actionCustom;
protected:
    void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
