#include "sboard.h"
#include <QPainter>

SBoard::SBoard(QWidget *parent)
    : QWidget{parent}
{
    connect(this, SIGNAL(clicked()), this, SLOT(mouseClicked()));
    initBoard();

}

void SBoard::initBoard() {
    guiState = GUIState::Playing;
    states.resize(8), valid.resize(8);
    for (int i = 0; i < 8; i++) {
        states[i].resize(8), valid[i].resize(8);
        for (int j = 0; j < 8; j++) states[i][j] = valid[i][j] = PositionState::None;
    }
    states[3][4] = states[4][3] = PositionState::Player1;
    states[3][3] = states[4][4] = PositionState::Player2;
    currentPlayer = PositionState::Player1;
    UpdateValid(states, valid, currentPlayer);
    update();
}
void SBoard::resizeEvent(QResizeEvent *ev) {
    printf("resized\n");
    update();
}

void SBoard::paintEvent(QPaintEvent *event) {
    QPainter _qpainter = QPainter(this);
    // calculate the size
    sboardHeight = fmin(this->height() - 50, 600);
    sboardWidth = fmin(this->width(), 600);
    sboardX = fmax(0, (this->height() - sboardHeight - 50) >> 1) + 50, sboardY = fmax((this->width() - sboardWidth) >> 1, 0);
    // set Color
    static QColor
        // the color of plaids
        _col0 = QColor(0xff, 0xff, 0xff), _col1 = QColor(0xff, 0, 0), _col2 = QColor(0, 0, 0xff),
        _coliv = QColor(0xcc,0xcc, 0xcc), _colv = QColor(0xee, 0xee, 0),
        // the color of text
        _colt = QColor(0, 0, 0);
    static char _text[105];
    plaidHeight = sboardHeight >> 3, plaidWidth = sboardWidth >> 3;

    if (guiState == GUIState::Playing) {
        // draw the state
        // format: current player, score 1 : score 2
        auto _result = GetResult(states);
        _qpainter.setPen(_colt);
        sprintf(_text, "Active Player : %d\n%d : %d", (int)currentPlayer, std::get<1>(_result), std::get<2>(_result));
        _qpainter.drawText(0, 0, this->width(), 50, Qt::AlignCenter, _text);
        auto _drawrect = [&_qpainter](const QColor &_col, const QRect &_rect) {
            _qpainter.setPen(_col);
            _qpainter.setBrush(_col);
            _qpainter.drawRect(_rect);
        };
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (valid[i][j] != PositionState::None)
                    _drawrect(_colv, QRect(sboardY + j * plaidWidth, sboardX + i * plaidHeight, plaidWidth, plaidHeight));
                else _drawrect(_coliv, QRect(sboardY + j * plaidWidth, sboardX + i * plaidHeight, plaidWidth, plaidHeight));
                switch (states[i][j]) {
                case PositionState::None:
                    _qpainter.setPen(_col0);
                    _qpainter.setBrush(_col0);
                    break;
                case PositionState::Player1:
                    _qpainter.setPen(_col1);
                    _qpainter.setBrush(_col1);
                    break;
                case PositionState::Player2:
                    _qpainter.setPen(_col2);
                    _qpainter.setBrush(_col2);
                    break;
                }
                _qpainter.drawRect(QRect(sboardY + j * plaidWidth + 5, sboardX + i * plaidHeight + 5, plaidWidth - 10, plaidHeight - 10));

            }
        }
    } else {
        _qpainter.setPen(_colt);
        sprintf(_text, "Winner : %d", (int)std::get<0>(lastResult));
        _qpainter.drawText(0, 0, this->width(), 50, Qt::AlignCenter, _text);
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
