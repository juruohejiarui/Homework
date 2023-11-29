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
    data[0][0] = 1;
    (rand() & 1 ? data[0][this->column - 1] : data[this->row - 1][0]) = 1;
}

bool GameState::checkValidLeft() {
    bool _fail = true;
    for (int i = 0; i < this->row; i++) {
        // condition 1 : have empty plaid between tiles
        int _p1 = 0, _p2 = this->column - 1;
        while (_p1 < this->column && data[i][_p1]) _p1++;
        while (_p2 >= 0 && data[i][_p2] == 0) _p2--;
        if (_p1 < _p2) { _fail = false; break; }
        // condition 2: the neighbour tiles can combine
        for (int j = 0; j < this->column - 1; j++)
            if (data[i][j] && data[i][j] == data[i][j + 1]) {
                _fail = false; break;
            }
        if (!_fail) break;
    }
    return !_fail;
}

bool GameState::checkValidRight() {
    bool _fail = true;
    for (int i = 0; i < this->row; i++) {
        int _p1 = 0, _p2 = this->column - 1;
        while (_p1 < this->column && data[i][_p1] == 0) _p1++;
        while (_p2 >= 0 && data[i][_p2]) _p2--;
        if (_p1 < _p2) { _fail = false; break; }

        for (int j = 0; j < this->column - 1; j++)
            if (data[i][j] && data[i][j] == data[i][j + 1]) {
                _fail = false; break;
            }
        if (!_fail) break;
    }
    return !_fail;
}

bool GameState::checkValidUp() {
    bool _fail = true;
    for (int i = 0; i < this->column; i++) {
        int _p1 = 0, _p2 = this->row - 1;
        while (_p1 < this->row && data[_p1][i]) _p1++;
        while (_p2 >= 0 && data[_p2][i] == 0) _p2--;
        if (_p1 < _p2) { _fail = false; break; }

        for (int j = 0; j < this->row - 1; j++)
            if (data[j][i] && data[j][i] == data[j + 1][i]) {
                _fail = false; break;
            }
        if (!_fail) break;
    }
    return !_fail;
}

bool GameState::checkValidDown() {
    bool _fail = true;
    for (int i = 0; i < this->column; i++) {
        int _p1 = 0, _p2 = this->row - 1;
        while (_p1 < this->row && data[_p1][i] == 0) _p1++;
        while (_p2 >= 0 && data[_p2][i]) _p2--;
        if (_p1 < _p2) { _fail = false; break; }

        for (int j = 0; j < this->row - 1; j++)
            if (data[j][i] && data[j][i] == data[j + 1][i]) {
                _fail = false; break;
            }
        if (!_fail) break;
    }
    return !_fail;
}
bool GameState::checkValid(GameOperation _o) {
    switch (_o) {
    case GameOperation::Left:   return checkValidLeft();
    case GameOperation::Right:  return checkValidRight();
    case GameOperation::Down:   return checkValidDown();
    case GameOperation::Up:     return checkValidUp();
    }
}

void GameState::updateStateLeft() {
    for (int i = 0; i < this->row; i++) {
        int _p = 0;
        // combine the valid pairs of plaid
        for (int j = 0; j < this->column - 1; j++) {
            int _p = j + 1;
            while (_p < this->column && data[i][_p] == 0) _p++;
            if (_p < this->column && data[i][j] == data[i][_p])
                data[i][j]++, score += (1 << data[i][j]), data[i][_p] = 0;
            j = _p - 1;
        }
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
        for (int j = this->column - 1; j > 0; j--) {
            int _p = j - 1;
            while (_p >= 0 && data[i][_p] == 0) _p--;
            if (_p >= 0 && data[i][j] == data[i][_p])
                data[i][j]++, score += (1 << data[i][j]), data[i][_p] = 0;
            j = _p + 1;
        }
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
        for (int j = 0; j < this->row - 1; j++) {
            int _p = j + 1;
            while (_p < this->row && data[_p][i] == 0) _p++;
            if (_p < this->row && data[j][i] == data[_p][i])
                data[j][i]++, score += (1 << data[j][i]), data[_p][i] = 0;
            j = _p - 1;
        }
        // move all the plaid to the top
        for (int j = 0; j < this->row; j++)
            if (data[j][i] > 0) data[_p++][i] = data[j][i];
        // clean
        for (int j = _p; j < this->row; j++) data[j][i] = 0;
    }
}

void GameState::updateStateDown() {
    for (int i = 0; i < this->column; i++) {
        int _p = this->row - 1;
        // combine the valid pairs of plaid
        for (int j = this->row - 1; j > 0; j--) {
            int _p = j - 1;
            while (_p >= 0 && data[_p][i] == 0) _p--;
            if (_p >= 0 && data[j][i] == data[_p][i])
                data[j][i]++, score += (1 << data[j][i]), data[_p][i] = 0;
            j = _p + 1;
        }
        // move all the plaid to the top
        for (int j = this->row - 1; j >= 0; j--)
            if (data[j][i] > 0) data[_p--][i] = data[j][i];
        // clean
        for (int j = _p; j >= 0; j--) data[j][i] = 0;
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
    for (int i = 0; i < this->row; i++)
        for (int j = 0; j < this->column; j++)
            printf("%d%c", data[i][j], (j == this->column - 1 ? '\n' : ' '));
    static std::vector< std::pair<int, int> > _empty_plaid;
    _empty_plaid.clear();
    for (int i = 0; i < this->row; i++)
        for (int j = 0; j < this->column; j++) if (data[i][j] == 0)
                _empty_plaid.push_back(std::make_pair(i, j));
    int _pos = rand() % _empty_plaid.size(), _data = (rand() % 10 == 0 ? 2 : 1);
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
