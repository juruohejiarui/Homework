#include "gameboard.h"
#include <QColor>
#include <QRect>
#include <QPainter>
#include <QFontDatabase>
#include <QMessageBox>
#include <QInputDialog>
#include <fstream>
#include <iostream>

#pragma region Message Box
bool ensureAbort() {
    QMessageBox msgbx;
    msgbx.setText("This operation will abort this game, continue?");
    msgbx.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    msgbx.setDefaultButton(QMessageBox::Ok);
    int res = msgbx.exec();
    if (res == QMessageBox::Ok) return true;
    else return false;
}
std::pair<bool, std::string> changeUserName(GameBoard *_parent) {
    bool _bret = false;
    QString _user_name = "";
    while (_user_name.length() == 0 || _user_name.length() < 20) {
        _user_name = QInputDialog::getText(
            _parent, "New User", "Input the new user name:", QLineEdit::Normal, "Default User", &_bret);
        if (!_bret) return std::make_pair(0, "");
        if (!_user_name.isEmpty() && _user_name.length() < 20) break;
    }
    return std::make_pair(true, _user_name.toStdString());
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
        tileFont = QFont(fontName, 14);
        textFont = QFont(fontName, 16);
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

#pragma region Event Handler
void GameBoard::mousePressEvent(QMouseEvent *ev) {
    mousePos = QPoint(ev->x(), ev->y());
}


void GameBoard::mouseReleaseEvent(QMouseEvent *ev) {
    if(mousePos == QPoint(ev->x(), ev->y())) emit clicked();
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


void GameBoard::initState() {
    configuration.initState();
}

bool GameBoard::tryNewGame() {
    if (configuration.getStatePackage().getCurrentState().end() && !ensureAbort())
        return false;
    initState();
    switchView(GUIState::Playing);
}

void GameBoard::changeTheme(const std::string &_path) {
    configuration.setThemePath(_path);

    std::ifstream ifs(configuration.getThemePath(), std::ios::binary);
    auto _readint = [&ifs]() -> int {
        int dt; ifs.read((char *)&dt, sizeof(int));
        printf("dt = %8x\n", dt);
        return dt;
    };
    backgroundColor = _readint();
    textColor = _readint();
    tileTextColor = _readint();
    for (int i = 0; i < 15; i++) tileColor[i] = _readint();

    ifs.close();
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
    if (_key == Qt::Key_Escape) {
        switchView(GUIState::End);
    }
    else if (_key == Qt::Key_Z) configuration.getStatePackage().undo();
    else {
        if (configuration.getStatePackage().getCurrentState().checkValid(getOperation(_key)))
            configuration.getStatePackage().Operate(getOperation(_key));
    }
    update();
}

void GameBoard::keyHandler_End(int _key) {
    if (_key == Qt::Key_Z) {
        if (configuration.getStatePackage().getCurrentState().end() || ensureAbort()) {
            // update the rank list
            configuration.updateRankList(
                GameResult(
                    configuration.getStatePackage().getCurrentState().getScore(),
                    time(NULL),
                    std::make_pair(configuration.getRow(), configuration.getColumn()),
                    configuration.getPlayer())
                );
            initState();
            configuration.save();
            switchView(GUIState::Playing);
        }
    } else if (_key == Qt::Key_Escape) {
        switchView(GUIState::Playing);
    } else {
        // update the rank list
        configuration.updateRankList(
            GameResult(
                configuration.getStatePackage().getCurrentState().getScore(), 
                time(NULL),
                std::make_pair(configuration.getRow(), configuration.getColumn()),
                configuration.getPlayer())
        );
        configuration.save();
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
    
}
#pragma endregion

#pragma region GUI update
void GameBoard::switchView(GUIState _gui_state) {
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
    int _topbar_h = 50, _board_w, _board_h, _plaid_h, _plaid_w, _tile_h, _tile_w, _board_x, _board_y, _tile_x, _tile_y;
    // use the width of this board
    if (this->width() * _row > (this->height() - _topbar_h) * _column)
        _board_h = std::min((this->height() - _topbar_h), 500),
            _board_w = _board_h * _column / _row;
    else
        _board_w = std::min(this->width(), 500),
            _board_h = _board_w * _row / _column;

    _plaid_h = _board_h / _row, _plaid_w = _board_w / _column;
    _tile_h = _plaid_h - ceil(_plaid_h / 10.0), _tile_w = _plaid_w - ceil(_plaid_w / 10.0);

    _board_x = (this->width() - _board_w) >> 1;
    _board_y = ((this->height() - _board_h - _topbar_h) >> 1) + _topbar_h;
    _tile_x = ceil(_plaid_w / 10.0), _tile_y = ceil(_plaid_h / 10.0);

    // draw the background
    drawRectangle(_ptr, QRect(0, 0, this->width(), this->height()), backgroundColor);
    // draw the topbar
    sprintf(textBuffer, "Player : %s", configuration.getPlayer().c_str());
    drawText(_ptr, QRect(0, 0, this->width() >> 1, _topbar_h), textColor, textFont);
    sprintf(textBuffer, "Score\n %d", configuration.getStatePackage().getCurrentState().getScore());
    drawRectangle(_ptr, QRect(this->width() >> 1, 0, this->width() >> 1, _topbar_h), tileColor[14]);
    drawText(_ptr, QRect(this->width() >> 1, 0, this->width() >> 1, _topbar_h), textColor, textFont);

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
                drawText(_ptr, _r, tileTextColor, tileFont);
            }

        }
}

void GameBoard::updateGUI_End() {
    QPainter _ptr = QPainter(this);

    drawRectangle(_ptr, QRect(0, 0, this->width(), this->height()), backgroundColor);
    
    sprintf(textBuffer, "Player : %s", configuration.getPlayer().c_str());
    drawText(_ptr, QRect(0, 0, this->width(), 50), textColor, textFont);

    sprintf(textBuffer, "Score : %d", configuration.getStatePackage().getCurrentState().getScore());
    drawText(_ptr, QRect(0, 50, this->width(), 50), tileTextColor, textFont);

    sprintf(textBuffer, "Continue [Esc]");
    drawRectangle(_ptr, QRect(50, 100, this->width() - 100, 50), tileColor[0]);
    drawText(_ptr, QRect(50, 100, this->width() - 100, 50), tileTextColor, textFont);

    sprintf(textBuffer, "Start a New Game [Z]");
    drawRectangle(_ptr, QRect(50, 155, this->width() - 100, 50), tileColor[0]);
    drawText(_ptr, QRect(50, 155, this->width() - 100, 50), tileTextColor, textFont);

    sprintf(textBuffer, "Show Rank List [Arrow]");
    drawRectangle(_ptr, QRect(50, 210, this->width() - 100, 50), tileColor[0]);
    drawText(_ptr, QRect(50, 210, this->width() - 100, 50), tileTextColor, textFont);
}
void GameBoard::updateGUI_RankList() {
    QPainter _ptr = QPainter(this);
    drawRectangle(_ptr, QRect(0, 0, this->width(), this->height()), backgroundColor);
    sprintf(textBuffer, "Back [Esc]");
    drawText(_ptr, QRect(0, 0, this->width() >> 1, 50), textColor, textFont);
    sprintf(textBuffer, "Scroll or [Arrow]");
    drawText(_ptr, QRect(this->width() >> 1, 0, this->width() >> 1, 50), textColor, tileFont);

    int _item_width = (this->width() - 70) / 3, _record_height = (this->height() - 50) / 10;
    for (int i = 0; i < 10 && i + scrollPosition < configuration.getRankList().size(); i++) {
        drawRectangle(_ptr, QRect(0, i * _record_height + 50, this->width(), _record_height), tileColor[14]);
        sprintf(textBuffer, "%d", i + scrollPosition + 1);
        drawText(_ptr, QRect(0, i * _record_height + 50, 50, _record_height), textColor, tileFont);

        auto &_record = configuration.getRankList()[i + scrollPosition];
        sprintf(textBuffer, "%s", _record.player.c_str());
        drawText(_ptr, QRect(50, i * _record_height + 50, _item_width, _record_height), textColor, tileFont);

        sprintf(textBuffer, "%d", _record.score);
        drawText(_ptr, QRect(50 + _item_width, i * _record_height + 50, _item_width, _record_height), textColor, tileFont);

        memcpy(textBuffer, std::ctime(&_record.time), 100);
        textBuffer[strlen(textBuffer) - 1] = '\0';
        drawText(_ptr, QRect(50 + (_item_width << 1), i * _record_height + 50, _item_width + 20, _record_height), textColor, tileFont);
    }
}
#pragma endregion

