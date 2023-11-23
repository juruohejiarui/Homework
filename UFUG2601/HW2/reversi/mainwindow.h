#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QAction>
#include <sboard.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

protected:
    void resizeEvent(QResizeEvent *ev) Q_DECL_OVERRIDE;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void newGameItemClicked();
    void changeSize8_8ItemClicked();
    void changeSize8_16ItemClicked();
    void changeSize16_16ItemClicked();
private:
    Ui::MainWindow *ui;
    QWidget *centralWidget;
    SBoard *sboard;

    QAction *newGameItem;

    QAction *changeSize8_8Item, *changeSize8_16Item, *changeSize16_16Item;
};
#endif // MAINWINDOW_H
