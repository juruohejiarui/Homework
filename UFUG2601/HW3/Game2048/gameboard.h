#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include <QWidget>
#include <QKeyEvent>
#include <stack>
#include <vector>
#include "gamestate.h"

#include "configuration.h"

enum class GUIState {
    Playing, End, RankList,
};

class GameBoard : public QWidget
{
    Q_OBJECT
private:
    void initState();
    int currentPressedKey;

    int scrollPosition;

    Configuration configuration;
    GUIState currentView;

    void initBoard();

    void updateGUI_Playing();
    void updateGUI_RankList();
    void updateGUI_End();
    void updateGUI();

    void keyHandler_Playing(int _key);
    void keyHandler_RankList(int _key);
    void keyHandler_End(int _key);
    void keyHandler(int _key);


    // load the configure from the default path
    int loadConfig();

protected:
    virtual void keyPressEvent(QKeyEvent *ev);
    virtual void keyReleaseEvent(QKeyEvent *ev);

public:
    explicit GameBoard(QWidget *parent = nullptr);

    // set the default config path
    void setConfig(std::string _path);

    void switchView(GUIState _gui_state);
    GUIState getCurrentGUIState();

    // try to abort the current game and create a new game
    bool tryAbort();
    bool tryResizeBoard(int _row, int _col);

    int getRow();
    int getColumn();

    void Undo();
    void Operator(GameOperation _o);

signals:
};

#endif // GAMEBOARD_H
