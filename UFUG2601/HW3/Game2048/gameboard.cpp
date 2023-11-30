#include "gameboard.h"
#include "dialogs.h"
#include <QColor>
#include <QRect>
#include <QPainter>
#include <QFontDatabase>
#include <QMessageBox>
#include <QInputDialog>
#include <fstream>
#include <iostream>

#pragma region Message Box
std::pair<bool, std::string> changeUserName(GameBoard *_parent) {
    bool _bret = false;
    QString _user_name = "";
    _parent->releaseKeyboard();
    _user_name = QInputDialog::getText(
        _parent, "New User", "Input the new user name:", QLineEdit::Normal, "Default User", &_bret);
    _parent->grabKeyboard();
    if (!_bret) return std::make_pair(0, "");
    if (!_user_name.isEmpty() && _user_name.length() < 20)
        return std::make_pair(true, _user_name.toStdString());
    else {
        showWarning("Valid User Name", "The user name must not be empty and be in 20 characters.");
        return std::make_pair(false, "");
    }
}
std::pair<int, int> changeSize(GameBoard *_parent) {
    bool _bret = false, _bret2 = false;
    _parent->releaseKeyboard();
    QString _size_str = QInputDialog::getMultiLineText(_parent, "New Size", "Input the new Size\nLine 1 : Row\nLine 2 : Column", "4\n4", &_bret);
    QStringList _list = _size_str.split('\n');
    _parent->grabKeyboard();
    if (!_bret) return std::make_pair(0, 0);
    if (_list.length() < 2) {
        showWarning("Valid Content", "The input for the new size is invalid\n");
        return std::make_pair(0, 0);
    }
    std::pair<int, int> _res = std::make_pair(_list[0].toInt(&_bret), _list[1].toInt(&_bret2));
    if (!_bret || !_bret2 || _res.first <= 0 || _res.first > 8 || _res.second <= 0 || _res.second > 8) {
        showWarning("Valid Content", "The new size is invalid\nnumbers of row and column must be in [1, 8]\n");
        return std::make_pair(0, 0);
    }
    return _res;
}

#pragma endregion

GameBoard::GameBoard(QWidget *parent)
    : QWidget{parent}
{

    connect(this, SIGNAL(clicked()), this, SLOT(mouseClicked()));
    this->grabKeyboard();
    this->setFocusPolicy(Qt::FocusPolicy::StrongFocus);
    this->setFocus();

    // set GUI information
    changeTheme(configuration.getThemePath());
    {
        int fontId = QFontDatabase::addApplicationFont(":/Resources/Fonts/GenshinFont.ttf");
        QString fontName = QFontDatabase::applicationFontFamilies(fontId).at(0);
        smallFont = QFont(fontName, 12);
        mediumFont = QFont(fontName, 14);
        LargeFont = QFont(fontName, 16);
    }
    switchView(GUIState::Playing);
}

inline int isValidKey(int _key) {
    return _key == Qt::Key_Left || _key == Qt::Key_Up || _key == Qt::Key_Down || _key == Qt::Key_Right
           || _key == Qt::Key_Escape || _key == Qt::Key_Z;
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


void GameBoard::initState() {
    configuration.initState();
}

bool GameBoard::tryNewGame() {
    if (!configuration.getStatePackage().getCurrentState().end() && !ensureAbort())
        return false;
    save();
    initState();
    switchView(GUIState::Playing);
    return true;
}

void GameBoard::undo() {
    configuration.getStatePackage().undo();
    update();
}

bool GameBoard::tryChangePlayer() {
    auto _res = changeUserName(this);
    if (!_res.first) return false;
    configuration.setPlayer(_res.second);
    update();
    return true;
}

bool GameBoard::tryResizeBoard(int _row, int _col) {
    if (!configuration.getStatePackage().getCurrentState().end() && !ensureAbort()) return false;
    configuration.setColumn(_col), configuration.setRow(_row);
    update();
    return true;
}

bool GameBoard::tryResizeBoard() {
    auto _res = changeSize(this);
    if (!_res.first || !_res.second) return false;
    return tryResizeBoard(_res.first, _res.second);
}

bool GameBoard::tryOperate(GameOperation _o) {
    if (!configuration.getStatePackage().getCurrentState().checkValid(_o)) return false;
    bool _res = configuration.getStatePackage().Operate(_o);
    if (_res) switchView(GUIState::End);
    update();
}

void GameBoard::changeTheme(const std::string &_path) {
    configuration.setThemePath(_path);

    std::ifstream ifs(configuration.getThemePath(), std::ios::binary);
    auto _readint = [&ifs]() -> int {
        int dt; ifs.read((char *)&dt, sizeof(int));
        return dt;
    };
    backgroundColor = _readint();
    textColor = _readint();
    tileTextColor = _readint();
    for (int i = 0; i < 15; i++) tileColor[i] = _readint();
    ifs.close();

    update();
}

void GameBoard::save() {
    configuration.updateRankList();
    configuration.save();
}

#pragma region Event Handler
void GameBoard::mousePressEvent(QMouseEvent *ev) {
    mousePos = QPoint(ev->x(), ev->y());
}

void GameBoard::mouseReleaseEvent(QMouseEvent *ev) {
    if(mousePos == QPoint(ev->x(), ev->y())) emit clicked();
    else {
        if (currentView == GUIState::Playing) {
            QPoint _delta = QPoint(ev->x(), ev->y()) - mousePos;
            if (abs(_delta.x()) < 5 || abs(_delta.y()) < 5) return ;
            GameOperation _o;
            if (std::abs(_delta.x()) > std::abs(_delta.y())) {
                if (_delta.x() < 0) _o = GameOperation::Left;
                else _o = GameOperation::Right;
            } else if (_delta.y() > 0) _o = GameOperation::Down;
            else _o = GameOperation::Up;

            tryOperate(_o);
        }
    }
}
void GameBoard::mouseClicked() {
    switch (currentView) {
    case GUIState::Playing:     mouseHandler_Playing();      return ;
    case GUIState::End:         mouseHandler_End();         return ;
    case GUIState::RankList:    mouseHandler_RankList();    return ;
    }
}

void GameBoard::paintEvent(QPaintEvent *event) {
    updateGUI();
}
void GameBoard::keyPressEvent(QKeyEvent *ev)
{
    if (isValidKey(ev->key())) currentPressedKey = ev->key();
    else QWidget::keyPressEvent(ev);
}

void GameBoard::keyReleaseEvent(QKeyEvent *ev)
{
    if (isValidKey(ev->key()) && ev->key() == currentPressedKey) {
        this->keyHandler(ev->key());
        return ;
    }
    currentPressedKey = ev->key();
    QWidget::keyReleaseEvent(ev);
}
#pragma endregion

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
    if (_key == Qt::Key_Escape) switchView(GUIState::End);
    else if (_key == Qt::Key_Z) configuration.getStatePackage().undo();
    else tryOperate(getOperation(_key));
    update();
}

void GameBoard::keyHandler_End(int _key) {
    if (_key == Qt::Key_Z) {
        tryNewGame();
    } else if (_key == Qt::Key_Escape) {
        switchView(GUIState::Playing);
    } else {
        switchView(GUIState::RankList);
    }
}

void GameBoard::keyHandler_RankList(int _key) {
    if (_key == Qt::Key_Escape) {
        switchView(GUIState::Playing);
    } else if (_key == Qt::Key_Up) {
        scrollPosition = std::max(scrollPosition - 1, 0);
    } else if (_key == Qt::Key_Down) {
        scrollPosition = std::max(std::min(scrollPosition + 1, (int)configuration.getRankList().size() - 10), 0);
    }
    update();
}

#pragma endregion

#pragma region Mouse Handler
void GameBoard::mouseHandler_Playing() {
    save();
    if (mousePos.x() < this->width() >> 1 && mousePos.y() < 50) tryChangePlayer();
    if (mousePos.x() > this->width() >> 1 && mousePos.y() < 50) switchView(GUIState::RankList);
}

void GameBoard::mouseHandler_RankList() {
    if (mousePos.x() < this->width() >> 1 && mousePos.y() < 50) switchView(GUIState::Playing);
}

void GameBoard::mouseHandler_End() {
    if (mousePos.x() < 50 || mousePos.x() > this->width() - 50) return ;
    if (mousePos.y() > 100 && mousePos.y() <= 150) switchView(GUIState::Playing);
    if (mousePos.y() > 155 && mousePos.y() <= 205) {
        save();
        if (configuration.getStatePackage().getCurrentState().end() || ensureAbort()) {
            initState();
            configuration.save();
            switchView(GUIState::Playing);
        }
    } else if (mousePos.y() > 210 && mousePos.y() < 260) {
        save();
        switchView(GUIState::RankList);
    }
}
#pragma endregion

#pragma region GUI update
void GameBoard::switchView(GUIState _gui_state) {
    if (currentView == GUIState::Playing) configuration.updateRankList(), configuration.save();
    currentView = _gui_state;
    scrollPosition = 0;
    update();
}


void GameBoard::updateGUI() {
    switch (currentView) {
    case GUIState::Playing:     updateGUI_Playing();    return ;
    case GUIState::End:         updateGUI_End();        return ;
    case GUIState::RankList:    updateGUI_RankList();   return ;
    }
}

static char textBuffer[1005];
void drawRectangle(QPainter &_painter, QRect _r, int _c) {
    QColor _col = QColor((_c >> 16) & 0xff, (_c >> 8) & 0xff, _c & 0xff);
    _painter.setBrush(_col), _painter.setPen(_col), _painter.drawRect(_r);
}
void drawText(QPainter &_painter, QRect _r, int _c, QFont &_f) {
    QColor _col = QColor((_c >> 16) & 0xff, (_c >> 8) & 0xff, _c & 0xff);
    _painter.setBrush(_col), _painter.setPen(_col), _painter.setFont(_f), _painter.drawText(_r, Qt::AlignCenter, textBuffer);
}

void GameBoard::updateGUI_Playing() {

    QPainter _ptr = QPainter(this);

    // calculate the size
    int _row = configuration.getRow(), _column = configuration.getColumn();
    int _topbar_h = 50, _hintbar_h = 25, _board_w, _board_h, _plaid_h, _plaid_w, _tile_h, _tile_w, _board_x, _board_y, _tile_x, _tile_y;
    // use the width of this board
    if (this->width() * _row > (this->height() - _topbar_h) * _column)
        _board_h = std::min((this->height() - _topbar_h - _hintbar_h), 500),
            _board_w = _board_h * _column / _row;
    else
        _board_w = std::min(this->width(), 500),
            _board_h = _board_w * _row / _column;

    _plaid_h = _board_h / _row, _plaid_w = _board_w / _column;
    _tile_h = _plaid_h - ceil(_plaid_h / 10.0), _tile_w = _plaid_w - ceil(_plaid_w / 10.0);

    _board_x = (this->width() - _board_w) >> 1;
    _board_y = ((this->height() - _board_h - _topbar_h - _hintbar_h) >> 1) + _topbar_h + _hintbar_h;
    _tile_x = ceil(_plaid_w / 10.0), _tile_y = ceil(_plaid_h / 10.0);

    // draw the background
    drawRectangle(_ptr, QRect(0, 0, this->width(), this->height()), backgroundColor);
    // draw the topbar
    sprintf(textBuffer, "Player : %s", configuration.getPlayer().c_str());
    drawRectangle(_ptr, QRect(0, 0, this->width() >> 1, _topbar_h), tileColor[0]);
    drawText(_ptr, QRect(0, 0, this->width() >> 1, _topbar_h), textColor, LargeFont);

    sprintf(textBuffer, "Score\n %d", configuration.getStatePackage().getCurrentState().getScore());
    drawRectangle(_ptr, QRect(this->width() >> 1, 0, this->width() >> 1, _topbar_h), tileColor[12]);
    drawText(_ptr, QRect(this->width() >> 1, 0, this->width() >> 1, _topbar_h), textColor, mediumFont);
    // draw the hint bar
    sprintf(textBuffer, "[ESC] Pause, [Z] Undo, [Arrow]/[Mouse] Operation");
    drawText(_ptr, QRect(0, _topbar_h, this->width(), _hintbar_h), tileTextColor, smallFont);
    // draw the board
    for (int i = 0; i < _row; i++)
        for (int j = 0; j < _column; j++) {
            auto _r = QRect(_board_x + _plaid_w * j + _tile_x, _board_y + _plaid_h * i + _tile_y,
                            _tile_w, _tile_h);
            drawRectangle(_ptr, _r,
                        tileColor[std::min(configuration.getStatePackage().getCurrentState()[i][j], 15)]);
            if (configuration.getStatePackage().getCurrentState()[i][j] > 0) {
                int _display_number = (1 << configuration.getStatePackage().getCurrentState()[i][j]);
                sprintf(textBuffer, "%d", _display_number);
                drawText(_ptr, _r, tileTextColor, (_display_number > 8192 ? smallFont : mediumFont));
            }

        }
}

void GameBoard::updateGUI_End() {
    QPainter _ptr = QPainter(this);

    drawRectangle(_ptr, QRect(0, 0, this->width(), this->height()), backgroundColor);
    
    sprintf(textBuffer, "Player : %s", configuration.getPlayer().c_str());
    drawText(_ptr, QRect(0, 0, this->width(), 50), textColor, LargeFont);

    sprintf(textBuffer, "Score : %d", configuration.getStatePackage().getCurrentState().getScore());
    drawText(_ptr, QRect(0, 50, this->width(), 50), tileTextColor, LargeFont);

    sprintf(textBuffer, "Continue [Esc]");
    drawRectangle(_ptr, QRect(50, 100, this->width() - 100, 50), tileColor[0]);
    drawText(_ptr, QRect(50, 100, this->width() - 100, 50), tileTextColor, LargeFont);

    sprintf(textBuffer, "Start a New Game [Z]");
    drawRectangle(_ptr, QRect(50, 155, this->width() - 100, 50), tileColor[0]);
    drawText(_ptr, QRect(50, 155, this->width() - 100, 50), tileTextColor, LargeFont);

    sprintf(textBuffer, "Show Rank List [Arrow]");
    drawRectangle(_ptr, QRect(50, 210, this->width() - 100, 50), tileColor[0]);
    drawText(_ptr, QRect(50, 210, this->width() - 100, 50), tileTextColor, LargeFont);
}
void GameBoard::updateGUI_RankList() {
    QPainter _ptr = QPainter(this);
    drawRectangle(_ptr, QRect(0, 0, this->width(), this->height()), backgroundColor);
    sprintf(textBuffer, "Back [Esc]");
    drawRectangle(_ptr, QRect(0, 0, this->width() >> 1, 50), tileColor[0]);
    drawText(_ptr, QRect(0, 0, this->width() >> 1, 50), textColor, LargeFont);
    sprintf(textBuffer, "Scroll or [Arrow]");
    drawText(_ptr, QRect(this->width() >> 1, 0, this->width() >> 1, 50), textColor, smallFont);

    int _item_width = (this->width() - 70) / 3, _record_height = (this->height() - 50) / 10;
    for (int i = 0; i < 10 && i + scrollPosition < configuration.getRankList().size(); i++) {
        drawRectangle(_ptr, QRect(0, i * _record_height + 50, this->width(), _record_height), tileColor[(i + scrollPosition < 3) ? 14 - (i + scrollPosition) : 8]);
        sprintf(textBuffer, "%d", i + scrollPosition + 1);
        drawText(_ptr, QRect(0, i * _record_height + 50, 50, _record_height), textColor, LargeFont);

        auto &_record = configuration.getRankList()[i + scrollPosition];
        sprintf(textBuffer, "%s", _record.player.c_str());
        drawText(_ptr, QRect(50, i * _record_height + 50, _item_width, _record_height), textColor, mediumFont);

        sprintf(textBuffer, "%d", _record.score);
        drawText(_ptr, QRect(50 + _item_width, i * _record_height + 50, _item_width, _record_height), textColor, LargeFont);

        memcpy(textBuffer, std::ctime(&_record.time), 100);
        textBuffer[strlen(textBuffer) - 1] = '\0';
        drawText(_ptr, QRect(50 + (_item_width << 1), i * _record_height + 50, _item_width + 20, _record_height), textColor, smallFont);
    }
}
#pragma endregion

