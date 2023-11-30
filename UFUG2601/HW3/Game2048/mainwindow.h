#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCloseEvent>
#include <QKeyEvent>
#include "gameboard.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
    GameBoard *gameBoard;
    QAction *actionResize33, *actionResize44, *actionResize55, *actionResize66, *actionResize88, *actionResizeCustom;
    QAction *actionNewGame,  *actionUndo;
    QAction *actionChangePlayer;
    QAction *actionThemeClassic, *actionThemeBlue;
protected:
    void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;
    void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void actionResize33Clicked();
    void actionResize44Clicked();
    void actionResize55Clicked();
    void actionResize66Clicked();
    void actionResize88Clicked();
    void actionResizeCustomClicked();

    void actionNewGameClicked();
    void actionUndoClicked();

    void actionChangePlayerClicked();

    void actionThemeClassicClicked();
    void actionThemeBlueClicked();
private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
