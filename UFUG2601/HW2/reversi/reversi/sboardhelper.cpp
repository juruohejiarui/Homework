#include "sboard.h"

const int offset_count = 8;
const int offset[8][2] = {{-1,0}, {1,0}, {0,1}, {0,-1}, {1,1}, {1,-1}, {-1,1}, {-1,-1}};

void SwitchPlayer(PositionState &_state) {
    if (_state == PositionState::Player1) _state = PositionState::Player2;
    else _state = PositionState::Player1;
}

// Check if this operation is valid
bool CheckValid(Board &_states, int _x, int _y, PositionState _player) {
    bool res = false;
    // the position must be empty
    if (_states[_x][_y] != PositionState::None) return false;
    // try all the directions
    for (int id = 0; id < offset_count; id++) {
        int px = _x + offset[id][0], py = _y + offset[id][1];
        while (px >= 0 && py >= 0 && px < _states.size() && py < _states[0].size()) {
            // if the position is empty, then this direction is invalid
            if (_states[px][py] == PositionState::None) break;
            if (_states[px][py] == _player) {
                res |= (fmax(abs(_x - px), abs(_y - py)) >= 2);
                printf("(%d, %d) <-> (%d, %d)\n", _x, px, _y, py);
                break;
            }
            px += offset[id][0], py += offset[id][1];
        }
        if (res) break;
    }
    return res;
}

void UpdateState(Board &_states, int _x, int _y, PositionState _player) {
    _states[_x][_y] = _player;
    // try every direction
    for (int id = 0; id < offset_count; id++) {
        int px = _x + offset[id][0], py = _y + offset[id][1];
        // the flag that shows whether this direction is valid
        bool valid = false;
        while (px >= 0 && px >= 0 && px < _states.size() && py < _states[0].size()) {
            // if the position is empty, then this direction is invalid
            if (_states[px][py] == PositionState::None) break;
            if (_states[px][py] == _player) {
                valid = true;
                break;
            }
            px += offset[id][0], py += offset[id][1];
        }
        if (valid) {
            px = _x + offset[id][0], py = _y + offset[id][1];
            while (_states[px][py] != _player) {
                _states[px][py] = _player;
                px += offset[id][0], py += offset[id][1];
            }
        }
    }
}

// return whether _player can operate
bool UpdateValid(Board &_states, Board &_valid, PositionState _player) {
    bool _res = false;
    for (int i = 0; i < _states.size(); i++)
        for (int j = 0; j < _states[0].size(); j++) {
            bool _subres = CheckValid(_states, i, j, _player);
            _res |= _subres;
            _valid[i][j] = (_subres ? _player : PositionState::None);
        }
    return _res;
}

GameResult GetResult(Board &_states) {
    int _count[3] = {0, 0, 0};
    for (int i = 0; i < _states.size(); i++) {
        for (int j = 0; j < _states[0].size(); j++) {
            _count[(int)_states[i][j]]++;
        }
    }
    return std::tuple((PositionState)(_count[1] > _count[2] ? 1 : 2), _count[1], _count[2]);
}
