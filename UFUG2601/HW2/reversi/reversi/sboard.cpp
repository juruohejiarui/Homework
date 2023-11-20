#include "sboard.h"
#include <QPainter>
#include <QMessageBox>

SBoard::SBoard(QWidget *parent)
    : QWidget{parent}
{
    connect(this, SIGNAL(clicked()), this, SLOT(mouseClicked()));
    this->Row = this->Column = 8;
    initBoard();

}

GUIState SBoard::GetGUIState() { return guiState; }

bool ensureAbort() {
    QMessageBox msgbx;
    msgbx.setText("This operation will abort this game, continue?");
    msgbx.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    msgbx.setDefaultButton(QMessageBox::Ok);
    int res = msgbx.exec();
    if (res == QMessageBox::Ok) return true;
    else return false;
}

void SBoard::initBoard() {
    guiState = GUIState::Playing;
    states.resize(this->Row), valid.resize(this->Row);
    for (int i = 0; i < this->Row; i++) {
        states[i].resize(this->Column), valid[i].resize(this->Column);
        for (int j = 0; j < this->Column; j++) states[i][j] = valid[i][j] = PositionState::None;
    }
    states[this->Row / 2 - 1][this->Column / 2] = states[this->Row / 2][this->Column / 2 - 1] = PositionState::Player1;
    states[this->Row / 2 - 1][this->Column / 2 - 1] = states[this->Row / 2][this->Column / 2] = PositionState::Player2;
    currentPlayer = PositionState::Player1;
    UpdateValid(states, valid, currentPlayer);
    update();
}


bool SBoard::tryCreateNewGame() {
    if (guiState == GUIState::Playing && !ensureAbort()) return false;
    initBoard();
    return true;
}

bool SBoard::trySetSize(int _row, int _column) {
    if (guiState == GUIState::Playing && !ensureAbort()) return false;
    this->Row = _row, this->Column = _column;
    SBoard::initBoard();
    return true;
}
bool SBoard::trySetSize(std::pair<int, int> _size) { return trySetSize(_size.first, _size.second); }
std::pair<int, int> SBoard::GetSize() { return std::make_pair(this->Row, this->Column); }

void SBoard::resizeEvent(QResizeEvent *ev) {
    printf("resized\n");
    update();
}

void SBoard::paintEvent(QPaintEvent *event) {
    QPainter _qpainter = QPainter(this);
    // calculate the size
    if ((this->height() - 50) * this->Column < this->width() * this->Row)
        sboardHeight = this->height() - 50, sboardWidth = sboardHeight * this->Column / this->Row;
    else sboardWidth = this->width(), sboardHeight = sboardWidth * this->Row / this->Column;
    sboardX = fmax(0, (this->height() - sboardHeight - 50) >> 1) + 50, sboardY = fmax((this->width() - sboardWidth) >> 1, 0);
    // set Color
    static const QColor
        // the color of plaids
        _col0 = QColor(0xff, 0xff, 0xff), _col1 = QColor(0xff, 0, 0), _col2 = QColor(0, 0, 0xff),
        _coliv = QColor(0xcc,0xcc, 0xcc), _colv = QColor(0xee, 0xee, 0),
        // the color of text
        _colt = QColor(0, 0, 0);
    static char _text[105];
    plaidHeight = sboardHeight / this->Row, plaidWidth = sboardWidth / this->Column;

    if (guiState == GUIState::Playing) {
        // draw the state
        // format: current player, score 1 : score 2
        auto _result = GetResult(states);
        if (currentPlayer == PositionState::Player1)
            _qpainter.setPen(_col1);
        else _qpainter.setPen(_col2);
        sprintf(_text, "Active Player : %d\n", (int)currentPlayer);
        _qpainter.drawText(0, 0, this->width(), 50, Qt::AlignCenter, _text);

        _qpainter.setPen(_colt);
        sprintf(_text, "\n%d : %d", std::get<1>(_result), std::get<2>(_result));
        _qpainter.drawText(0, 0, this->width(), 50, Qt::AlignCenter, _text);
        auto _drawrect = [&_qpainter](const QColor &_col, const QRect &_rect) {
            _qpainter.setPen(_col);
            _qpainter.setBrush(_col);
            _qpainter.drawRect(_rect);
        };

        const int _board_size = fmin(2.0, ceil(plaidWidth * 0.1));

        for (int i = 0; i < this->Row; i++) {
            for (int j = 0; j < this->Column; j++) {
                const QRect _inner_rect = QRect(sboardY + j * plaidWidth + _board_size, sboardX + i * plaidHeight + _board_size,
                                                plaidWidth - 2 * _board_size, plaidHeight - 2 * _board_size),
                            _prompt_rect = QRect(sboardY + j * plaidWidth + _board_size * 2, sboardX + i * plaidHeight + _board_size * 2,
                                                 plaidWidth - _board_size * 4, plaidHeight - _board_size * 4),
                            _chess_rect = QRect(sboardY + j * plaidWidth + _board_size * 4, sboardX + i * plaidHeight + _board_size * 4,
                                                plaidWidth - _board_size * 8, plaidHeight - _board_size * 8);
                _drawrect(_coliv, QRect(sboardY + j * plaidWidth, sboardX + i * plaidHeight, plaidWidth, plaidHeight));
                if (valid[i][j] != PositionState::None)
                    _drawrect((currentPlayer == PositionState::Player1 ? _col1 : _col2), _inner_rect);
                else _drawrect(_col0, _inner_rect);
                _qpainter.setPen(_col0);
                _qpainter.setBrush(_col0);
                _qpainter.drawRect(_prompt_rect);
                switch (states[i][j]) {
                case PositionState::None:
                    break;
                case PositionState::Player1:
                    _qpainter.setPen(_col1);
                    _qpainter.setBrush(_col1);
                    _qpainter.drawEllipse(_chess_rect);

                    break;
                case PositionState::Player2:
                    _qpainter.setPen(_col2);
                    _qpainter.setBrush(_col2);
                    _qpainter.drawEllipse(_chess_rect);

                    break;
                }

            }
        }
    } else {
        if (std::get<0>(lastResult) == PositionState::Player1)
            _qpainter.setPen(_col1);
        else _qpainter.setPen(_col2);
        sprintf(_text, "Winner : %d", (int)std::get<0>(lastResult));
        _qpainter.drawText(0, 0, this->width(), 50, Qt::AlignCenter, _text);
        _qpainter.setPen(_colt);
        sprintf(_text, "Score : %d : %d", std::get<1>(lastResult), std::get<2>(lastResult));
        _qpainter.drawText(0, 50, this->width(), 50, Qt::AlignCenter, _text);
        sprintf(_text, "Click this window to start a new game");
        _qpainter.drawText(0, 100, this->width(), 50, Qt::AlignCenter, _text);
    }
}

void SBoard::mouseClicked() {
    if (guiState == GUIState::Playing) {
        // calculate the position
            if (sboardX > mousePos.y() || sboardY > mousePos.x()
                || sboardX + sboardHeight <= mousePos.y() || sboardY + sboardWidth <= mousePos.x())
            return ;
        plaidPos = QPoint((mousePos.y() - sboardX) / plaidHeight, (mousePos.x() - sboardY) / plaidWidth);
        printf("%d %d\n", plaidPos.x(), plaidPos.y());
        if (CheckValid(states, plaidPos.x(), plaidPos.y(), currentPlayer)) {
            UpdateState(states, plaidPos.x(), plaidPos.y(), currentPlayer);
        SwitchPlayer(currentPlayer);
        if (!UpdateValid(states, valid, currentPlayer)) {
            // switch plaer again
            SwitchPlayer(currentPlayer);
            if (!UpdateValid(states, valid, currentPlayer)) {
                    lastResult = GetResult(states);
                    guiState = GUIState::End;
                }
            }
        }
    } else {
        initBoard();
    }
    update();
}

void SBoard::mousePressEvent(QMouseEvent *ev) {
    mousePos = QPoint(ev->x(), ev->y());
}


void SBoard::mouseReleaseEvent(QMouseEvent *ev) {
    if(mousePos == QPoint(ev->x(), ev->y())) emit clicked();
}
