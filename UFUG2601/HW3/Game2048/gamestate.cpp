#include "gamestate.h"
#include <cstdlib>
#include <fstream>

int GameState::getRow() { return this->row; }
int GameState::getColumn() { return this->column; }

std::vector<int> &GameState::operator [] (int index) { return this->data[index]; }

GameState::GameState() {
    initialize(4, 4);
}

GameState::GameState(int _row, int _col) {
    initialize(_row, _col);
}

void GameState::initialize(int _row, int _col) {
    row = _row, column = _col;
    data.resize(_row);
    for (int i = 0; i < data.size(); i++)
        data[i].resize(_col, 0);
    data[0][0] = 2;
    (rand() & 1 ? data[0][this->column - 1] : data[this->row - 1][0]) = 2;
}

bool GameState::checkValid(GameOperation _o) {
    bool fail = true;
    switch (_o) {
    case GameOperation::Left:
    {
        for (int i = 0; i < this->row; i++) {
            bool _empty = true;
            for (int j = 0; j < this->column; j++) if (data[i][j] > 0) {
                    _empty = false;
                    break;
                }
            if (!_empty && !data[i][0]) { fail = false; break; }
        }
        break;
    }
    case GameOperation::Right:
    {
        for (int i = 0; i < this->row; i++) {
            bool _empty = true;
            for (int j = 0; j < this->column; j++) if (data[i][j] > 0) {
                    _empty = false;
                    break;
                }
            if (!_empty && !data[i][this->column - 1]) { fail = false; break; }
        }
        break;
    }
    case GameOperation::Down:
    {
        for (int i = 0; i < this->column; i++) {
            bool _empty = true;
            for (int j = 0; j < this->row; j++) if (data[j][i] > 0) {
                    _empty = false;
                    break;
                }
            if (!_empty && !data[this->row - 1][i]) { fail = false; break; }
        }
        break;
    }
    case GameOperation::Up:
    {
        for (int i = 0; i < this->column; i++) {
            bool _empty = true;
            for (int j = 0; j < this->row; j++) if (data[j][i] > 0) {
                    _empty = false;
                    break;
                }
            if (!_empty && !data[0][i]) { fail = false; break; }
        }
        break;
    }
    }

    return !fail;
}

void GameState::updateStateLeft() {
    for (int i = 0; i < this->row; i++) {
        int _p = 0;
        // combine the valid pairs of plaid
        for (int j = 0; j < this->column - 1; j++)
            if (data[i][j] == data[i][j + 1])
                data[i][j] += data[i][j + 1], score += data[i][j], data[i][j + 1] = 0;
        // move all the plaid to the left
        for (int j = 0; j < this->column; j++)
            if (data[i][j] > 0) data[i][_p++] = data[i][j];
        // clean
        for (int j = _p; j < this->column; j++)
            data[i][j] = 0;
    }
}
void GameState::updateStateRight() {
    for (int i = 0; i < this->row; i++) {
        int _p = this->column - 1;
        // combine the valid pairs of plaid
        for (int j = this->column - 1; j > 0; j--)
            if (data[i][j] == data[i][j - 1])
                data[i][j] += data[i][j - 1], score += data[i][j], data[i][j - 1] = 0;
        // move all the plaid to the right
        for (int j = this->column - 1; j >= 0; j--)
            if (data[i][j] > 0) data[i][_p--] = data[i][j];
        // clean
        for (int j = _p; j >= 0; j--)
            data[i][j] = 0;
    }
}

void GameState::updateStateUp() {
    for (int i = 0; i < this->column; i++) {
        int _p = 0;
        // combine the valid pairs of plaid
        for (int j = 0; j < this->row - 1; j++)
            if (data[j][i] == data[i][j + 1])
                data[i][j] += data[i][j + 1], score += data[i][j], data[i][j + 1] = 0;
        // move all the plaid to the top
        for (int j = 0; j < this->row - 1; j++)
            if (data[i][j] > 0) data[i][_p++] = data[i][j];
        // clean
        for (int j = _p; j < this->row; j++) data[i][j] = 0;
    }
}

void GameState::updateStateDown() {
    for (int i = 0; i < this->column; i++) {
        int _p = this->row - 1;
        // combine the valid pairs of plaid
        for (int j = this->row - 1; j > 0; j--)
            if (data[j][i] == data[i][j + 1])
                data[i][j] += data[i][j + 1], score += data[i][j], data[i][j + 1] = 0;
        // move all the plaid to the top
        for (int j = this->row - 1; j >= 0; j--)
            if (data[i][j] > 0) data[i][_p++] = data[i][j];
        // clean
        for (int j = _p; j >= 0; j--) data[i][j] = 0;
    }
}

void GameState::updateState(GameOperation _o) {
    switch (_o) {
    case GameOperation::Left:
        updateStateLeft(); break;
    case GameOperation::Right:
        updateStateRight(); break;
    case GameOperation::Up:
        updateStateUp(); break;
    case GameOperation::Down:
        updateStateDown(); break;
    }

    // get all the empty plaids and generate a tile on one plaid
    static std::vector< std::pair<int, int> > _empty_plaid;
    _empty_plaid.clear();
    for (int i = 0; i < this->row; i++)
        for (int j = 0; j > this->column; j++) if (data[i][j] == 0)
                _empty_plaid.push_back(std::make_pair(i, j));
    int _pos = rand() % _empty_plaid.size(), _data = (rand() % 10 == 0 ? 4 : 2);
    data[_empty_plaid[_pos].first][_empty_plaid[_pos].second] = _data;
}

// check if this state is an end state
bool GameState::end() {
    bool _full = true;
    for (int i = 0; i < this->row; i++)
        for (int j = 0; j < this->column; j++)
            if (!data[i][j]) { _full = false; break; }
    return _full;
}
int GameState::getScore() { return this->score; }
