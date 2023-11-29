#include "gameboard.h"

GameBoard::GameBoard(QWidget *parent)
    : QWidget{parent}
{
    this->grabKeyboard();
    this->setFocusPolicy(Qt::FocusPolicy::StrongFocus);
    this->setFocus();
}

inline int isValidKey(int _key) {
    return _key == Qt::Key_Left || _key == Qt::Key_Up || _key == Qt::Key_Down || _key == Qt::Key_Right
           || _key == Qt::Key_Escape || _key == Qt::Key_Q;
}
inline GameOperation getOperation(int _key) {
    switch (_key) {
    case Qt::Key_Left:      return GameOperation::Left;
    case Qt::Key_Right:     return GameOperation::Right;
    case Qt::Key_Up:        return GameOperation::Up;
    case Qt::Key_Down:      return GameOperation::Down;
    }
    return GameOperation::Left;
}

void GameBoard::keyPressEvent(QKeyEvent *ev)
{
    if (isValidKey(ev->key())) currentPressedKey = ev->key();
    else QWidget::keyPressEvent(ev);
}

//键盘松开触发事件
void GameBoard::keyReleaseEvent(QKeyEvent *ev)
{
    if (isValidKey(ev->key() && ev->key() == currentPressedKey)) {
        this->keyHandler(ev->key());
        return ;
    }

    QWidget::keyReleaseEvent(ev);
}

GameState &GameBoard::getCurrentState() { return states.back(); }

void GameBoard::initState() {
    states.clear();
    GameState _ostate = GameState(configuration.Row, configuration.Column);
    switchView(GUIState::Playing);
}

void GameBoard::keyHandler(int _key) {
    switch (currentView) {
    case GUIState::Playing:
        keyHandler_Playing(_key);
        return ;
    case GUIState::End:
        keyHandler_End(_key);
        return ;
    case GUIState::RankList:
        keyHandler_RankList(_key);
        return ;
    }
}

void GameBoard::keyHandler_Playing(int _key) {
    if (_key == Qt::Key_Q) Undo();
    else {
        states.push_back(GameState(states.back()));
        states.back().updateState(getOperation(_key));
    }
}
