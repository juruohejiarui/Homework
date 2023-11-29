#include "gameboard.h"
#include <QColor>
#include <QRect>
#include <QPainter>
#include <QFontDatabase>
#include <QMessageBox>
#include <fstream>

bool ensureAbort() {
    QMessageBox msgbx;
    msgbx.setText("This operation will abort this game, continue?");
    msgbx.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    msgbx.setDefaultButton(QMessageBox::Ok);
    int res = msgbx.exec();
    if (res == QMessageBox::Ok) return true;
    else return false;
}

GameBoard::GameBoard(QWidget *parent)
    : QWidget{parent}
{

    connect(this, SIGNAL(clicked()), this, SLOT(mouseClicked()));
    this->grabKeyboard();
    this->setFocusPolicy(Qt::FocusPolicy::StrongFocus);
    this->setFocus();

    changeTheme(configuration.getThemePath());
    {
        int fontId = QFontDatabase::addApplicationFont(":/Resources/Fonts/GenshinFont.ttf");
        QString fontName = QFontDatabase::applicationFontFamilies(fontId).at(0);
        tileFont = QFont(fontName, 15);
        textFont = QFont(fontName, 16);
    }
    switchView(GUIState::Playing);
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

#pragma region Event Handler
void GameBoard::mousePressEvent(QMouseEvent *ev) {
    mousePos = QPoint(ev->x(), ev->y());
}


void GameBoard::mouseReleaseEvent(QMouseEvent *ev) {
    if(mousePos == QPoint(ev->x(), ev->y())) emit clicked();
}
void GameBoard::mouseClicked() {

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
        if (ensureAbort()) configuration.initState();
    }
    else if (_key == Qt::Key_Q) configuration.getStatePackage().undo();
    else {
        if (configuration.getStatePackage().getCurrentState().checkValid(getOperation(_key)))
            configuration.getStatePackage().Operate(getOperation(_key));
    }
    update();
}

void GameBoard::keyHandler_End(int _key) {
    if (_key == Qt::Key_Escape) {
        switchView(GUIState::Playing);
    }
    update();
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

void GameBoard::updateGUI_Playing() {
    char _text[105];

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

    auto _drawrect = [&_ptr](QRect _r, int _c) {
        QColor _col = QColor((_c >> 16) & 0xff, (_c >> 8) & 0xff, _c & 0xff);
        _ptr.setBrush(_col), _ptr.setPen(_col), _ptr.drawRect(_r);
    };
    auto _drawtext = [&_ptr, &_text](QRect _r, int _c, QFont _f) {
        QColor _col = QColor((_c >> 16) & 0xff, (_c >> 8) & 0xff, _c & 0xff);
        _ptr.setBrush(_col), _ptr.setPen(_col), _ptr.setFont(_f), _ptr.drawText(_r, Qt::AlignCenter, _text);
    };

    // draw the topbar
    _drawrect(QRect(0, 0, this->width(), this->height()), backgroundColor);
    sprintf(_text, "Player : %s", configuration.getPlayer().c_str());
    _drawtext(QRect(0, 0, this->width() >> 1, _topbar_h), textColor, textFont);
    sprintf(_text, "Score\n %d", configuration.getStatePackage().getCurrentState().getScore());
    _drawrect(QRect(this->width() >> 1, 0, this->width() >> 1, _topbar_h), tileColor[14]);
    _drawtext(QRect(this->width() >> 1, 0, this->width() >> 1, _topbar_h), textColor, textFont);

    printf("(%d, %d), w = %d, h = %d\n", _board_x, _board_y, _board_w, _board_h);
    // draw the board
    for (int i = 0; i < _row; i++)
        for (int j = 0; j < _column; j++) {
            auto _r = QRect(_board_x + _plaid_w * j - _tile_x, _board_y + _plaid_h * i + _tile_y,
                            _tile_w, _tile_h);
            _drawrect(  _r,
                        tileColor[std::min(configuration.getStatePackage().getCurrentState()[i][j], 13)]);
            if (configuration.getStatePackage().getCurrentState()[i][j] > 0) {
                int _display_number = (1 << configuration.getStatePackage().getCurrentState()[i][j]);
                sprintf(_text, "%d", _display_number);
                _drawtext(_r, tileTextColor, tileFont);
            }

        }
}

void GameBoard::updateGUI_End() {

}
void GameBoard::updateGUI_RankList() {
}
#pragma endregion

