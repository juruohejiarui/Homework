#include "gamestatepackage.h"
#include <fstream>

GameStatePackage::GameStatePackage()
{
}

GameStatePackage::GameStatePackage(const std::string &_path) {
    load(_path);
}

void GameStatePackage::load(const std::string &_path) {
    std::ifstream ifs(_path, std::ios::binary);
    filePath = _path;

    states.clear();

    if (!ifs.good()) {
        states.push_back(GameState());
        save();
        ifs.open(_path, std::ios::binary);
    }

    int _state_cnt, _row, _column, _temp;
    ifs.read((char *)&_state_cnt, sizeof(int));
    ifs.read((char *)&_row, sizeof(int));
    ifs.read((char *)&_column, sizeof(int));

    for(int i = 0; i < _state_cnt; i++) {
        auto _state = GameState(_row, _column);
        ifs.read((char *)&_temp, sizeof(int)), _state.setScore(_temp);
        printf("read score : %d\n", _temp);
        for (int x = 0; x < _row; x++)
            for (int y = 0; y < _column; y++)
                ifs.read((char *)&_state[x][y], sizeof(char));
        states.push_back(_state);
    }
}

void GameStatePackage::save() {
    std::ofstream ofs(filePath, std::ios::binary);

    int _temp = states.size();
    ofs.write((char *)&_temp, sizeof(int));
    _temp = states[0].getRow();
    ofs.write((char *)&_temp, sizeof(int));
    _temp = states[0].getColumn();
    ofs.write((char *)&_temp, sizeof(int));

    for (auto &state : states) {
        _temp = state.getScore();
        ofs.write((char *)&_temp, sizeof(int));
        for (int x = 0; x < state.getRow(); x++)
            for (int y = 0; y < state.getColumn(); y++)
                ofs.write((char *)&state[x][y], sizeof(char));
    }
}

GameState &GameStatePackage::getCurrentState() { return states.back(); }

void GameStatePackage::undo() {
    if (states.size() > 1) states.pop_back();
}

void GameStatePackage::init() {
    int _row = states[0].getRow(), _col = states[0].getColumn();
    while (!states.empty()) states.pop_back();
    states.push_back(GameState(_row, _col));
}

bool GameStatePackage::Operate(GameOperation _o) {
    states.push_back(GameState(states[states.size() - 1]));
    getCurrentState().updateState(_o);
    if (states.size() > maxStateQueueSize) states.pop_front();
    return getCurrentState().end();
}

int GameStatePackage::getRow() { return states[0].getRow(); }
int GameStatePackage::getColumn() { return states[0].getColumn(); }
void GameStatePackage::setRow(int _row) {
    int _old_col = states[0].getColumn();
    states.clear(), states.push_back(GameState(_row, _old_col));
}
void GameStatePackage::setColumn(int _col) {
    int _old_row = states[0].getRow();
    states.clear(), states.push_back(GameState(_old_row, _col));
}
