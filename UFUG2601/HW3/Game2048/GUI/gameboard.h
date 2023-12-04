#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include <QWidget>
#include <QKeyEvent>
#include <stack>
#include <vector>
#include "../Core/gamestate.h"

#include "../Core/configuration.h"

enum class GUIState {
    Playing, End, RankList,
};

class GameBoard : public QWidget
{
    Q_OBJECT
private:
    void initState();
    int currentPressedKey;

    #pragma region GUI Information
    int scrollPosition;
    QPoint mousePos;
    QFont smallFont, mediumFont, LargeFont;
    GUIState currentView;
    int tileColor[15], backgroundColor, boardColor, textColor, tileTextColor;
    #pragma endregion

    Configuration configuration;

    void updateGUI_Playing();
    void updateGUI_RankList();
    void updateGUI_End();
    void updateGUI();

    void keyHandler_Playing(int _key);
    void keyHandler_RankList(int _key);
    void keyHandler_End(int _key);
    void keyHandler(int _key);

    void mouseHandler_Playing();
    void mouseHandler_RankList();
    void mouseHandler_End();

    // load the configure from the default path
    int loadConfig();

protected:
    void wheelEvent(QWheelEvent *event) Q_DECL_OVERRIDE;
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *ev) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent *ev) Q_DECL_OVERRIDE;
    void keyPressEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;
    void keyReleaseEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;
public slots:
    void mouseClicked();
public:
    explicit GameBoard(QWidget *parent = nullptr);

    void changeTheme(const std::string &_path);
    void switchView(GUIState _gui_state);

    // try to abort the current game and create a new game
    bool tryNewGame();
    void undo();

    bool tryChangePlayer();
    bool tryResizeBoard(int _row, int _col);
    // use custom size
    bool tryResizeBoard();

    bool tryOperate(GameOperation _o);
    
    // set the default config path
    void setConfig(std::string _path);
    int getRow();
    int getColumn();

    void save();

signals:
    void clicked();
};

#endif // GAMEBOARD_H
