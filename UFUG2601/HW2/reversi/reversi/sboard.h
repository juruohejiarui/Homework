#ifndef SBOARD_H
#define SBOARD_H

#include <tuple>
#include <QWidget>
#include <QMouseEvent>
#include <QPaintEvent>

enum class PositionState {
    None, Player1, Player2
};

typedef std::vector< std::vector<PositionState> > Board;

void SwitchPlayer(PositionState &_state);
bool CheckValid(Board &_states, int _x, int _y, PositionState _player);
void UpdateState(Board &_states, int _x, int _y, PositionState _player);
bool UpdateValid(Board &_states, Board &_valid, PositionState _player);

typedef std::tuple<PositionState, int, int> GameResult;
GameResult GetResult(Board &_states);

class SBoard : public QWidget
{

    Q_OBJECT
public slots:
    void mouseClicked();
public:
    explicit SBoard(QWidget *parent = nullptr);
protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE; //绘图
    void mousePressEvent(QMouseEvent *ev) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent *ev) Q_DECL_OVERRIDE;
    void resizeEvent(QResizeEvent *ev) Q_DECL_OVERRIDE;
signals:
    void clicked();
private:
    QPoint mousePos, plaidPos;

    int sboardX, sboardY, sboardWidth, sboardHeight;
    int plaidWidth, plaidHeight;

    Board states, valid;

    PositionState currentPlayer;
    GameResult lastResult;

    enum class GUIState {
        Playing, End,
    } guiState;

    void initBoard();
};

#endif // SBOARD_H
