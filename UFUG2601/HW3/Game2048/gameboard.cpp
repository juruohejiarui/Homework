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


void GameBoard::initState() {
    configuration.initState();
}


#pragma region Key Handler
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
    if (_key == Qt::Key_Q) configuration.getStatePackage().undo();
    else {

    }
}

void GameBoard::keyHandler_End(int _key) {
    if (_key == Qt::Key_Escape) {
        switchView(GUIState::Playing);
    }
}

void GameBoard::keyHandler_RankList(int _key) {
    if (_key == Qt::Key_Escape) {
        switchView(GUIState::Playing);
    } else if (_key == Qt::Key_Up) {
        scrollPosition = std::max(scrollPosition - 1, 0);
    } else if (_key == Qt::Key_Down) {
        scrollPosition = std::max(scrollPosition + 1, (int)ceil(configuration.getRankList().size() / 10.0) - 1);
    }
    update();
}

#pragma endregion

#pragma region GUI update
void GameBoard::switchView(GUIState _gui_state) {
    currentView = _gui_state;
    update();
}

void GameBoard::updateGUI() {

}

#pragma endregion
