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
    QAction *actionNewGame, *actionHelp;
protected:
    void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void actionHelpClicked();
    void actionNewGameClicked();
private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
